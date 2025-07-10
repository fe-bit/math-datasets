
from typing import List, Dict
from typing_extensions import TypedDict
import re
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from sympy import sympify
from sympy.core.sympify import SympifyError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain.schema import messages_to_dict
import time


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str
    message: List

# Example: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
# Plan: First, calculate Tina's overtime wage. This is her regular wage plus half of her regular wage. #E1 = Calculator[18 + (18 / 2)]
# Plan: Next, determine how many overtime hours Tina works each day. She works 10 hours and is eligible for overtime after 8, so we need to find the difference. #E2 = Calculator[10 - 8]
# Plan: Now, calculate Tina's daily overtime pay. Multiply her overtime wage by the number of overtime hours she works each day. #E3 = Calculator[#E1 * #E2]
# Plan: Calculate Tina's regular pay each day. Multiply her regular hourly wage by the number of regular hours she works (8 hours). #E4 = Calculator[18 * 8]
# Plan: Calculate Tina's total daily pay. Add her regular pay and her overtime pay for each day. #E5 = Calculator[#E3 + #E4]
# Plan: Finally, calculate Tina's total earnings for the week. Multiply her total daily pay by the number of days she works (5 days). #E6 = Calculator[#E5 * 5]
class ReWOOModel:
    def __init__(self, model: ChatHuggingFace|ChatGoogleGenerativeAI|ChatOllama, sleep_time: int=10, with_examples: bool=True):
        self.model = model
        self.sleep_time = sleep_time
        prompt = ReWOOModel.get_prompt(with_examples=with_examples)

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        self.planner = prompt_template | self.model
    
    def __call__(self, task: str, silent:bool=True, wait=True) -> str:
        result = []
        app = self.get_graph()
        for s in app.stream({"task": task}):
            if not silent:
                print(s)
                print("---")
            result.append(s)
            time.sleep(self.sleep_time)
        return result

    def get_plan(self, state: ReWOO):
        task = state["task"]
        result = self.planner.invoke({"task": task})
        assistant_response = self.__extract_assistant_response(result.content)
        matches = ReWOOModel.parse_plan(assistant_response)
        return {"steps": matches, "plan_string": assistant_response, "message": messages_to_dict([result])}
    
    @classmethod
    def parse_plan(cls, plan: str) -> List[tuple]:
        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, plan)
        return matches
    
    def _get_current_task(self, state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO):
        """Worker node that executes the tools of a given plan."""
        _step = self._get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        _results = (state["results"] or {}) if "results" in state else {}
        
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        
        if tool == "Calculator":
            try:
                expr = tool_input
                for k, v in _results.items():
                    # Remove '#' and ensure variables are strings or numbers
                    expr = expr.replace(k, str(v))
                    expr = expr.replace(k.lstrip('#'), str(v))
                
                expr = expr.replace("\"", "").replace("'", "")  # Remove quotes if any
                # Parse and evaluate safely
                evaluated = sympify(expr).evalf()
                result = str(evaluated)
            except SympifyError as e:
                result = f"SymPy Error: {e}"
            except Exception as e:
                result = f"Error: {e}"
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}

    def solve(self, state: ReWOO):
        try:
            last_value = float(list(state["results"].values())[-1])
            return {"result": last_value}
        except (ValueError, IndexError):
            result = "Error: Unable to retrieve the final result from the steps."
            return {"result": result, "message": []}
    
    def _route(self, state):
        _step = self._get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"
    
    def get_graph(self):
        graph = StateGraph(ReWOO)
        graph.add_node("plan", self.get_plan)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)
        
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self._route)
        graph.add_edge(START, "plan")
        app = graph.compile()
        return app

    @classmethod
    def get_prompt(cls, with_examples: bool) -> str:
        prompt = """You are a helpful assistant that solves math word problems by breaking them down into small steps.  
    Each step has:
    - A Plan: a short explanation of what to calculate.
    - A Calculator call: an exact arithmetic expression to compute.

    Use the tool:
    Calculator[input] → solves arithmetic like "180 / 3", "12 * 4", or "300 - 125".

    Rules:
    - Do NOT give the final answer directly.
    - Think step by step.
    - Use one Calculator call per step.
    - Save results as variables like #E1, #E2, etc.

    Step format:
    Plan: [what to calculate and why]  
    #EX = Calculator[math expression]

    """
        if with_examples:
            prompt += """
    Example:

    Task: A train travels 180 km in 3 hours. What is its average speed?

    Plan: To find the average speed, divide the distance by the time.  
    #E1 = Calculator[180 / 3]
    """
        prompt += """

    Now solve the following:

    Task: {task}
    """
        return prompt



    def __extract_assistant_response(self, result: str) -> str:
        if "<|im_start|>assistant" in result:
            assistant_part = result.split("<|im_start|>assistant")[-1]
        else:
            assistant_part = result
        return assistant_part

class PlanGenerator:
    def __init__(self, model: ChatHuggingFace|ChatGoogleGenerativeAI, sleep_time: int=10, with_examples: bool=True):
        self.model = model
        prompt = ReWOOModel.get_prompt(with_examples=with_examples)
        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        self.planner = prompt_template | self.model
    
    def generate_plan(self, task: str) -> str:
        result = self.planner.invoke({"task": task}).content
        assistant_response = self.__extract_assistant_response(result)
        return assistant_response

    def generate_plan_batch(self, tasks: list[str]) -> list[str]:
        try:
            inputs = [{"task": task} for task in tasks]
            results = self.planner.batch(inputs)
            return [self.__extract_assistant_response(r.content) for r in results]
        except Exception as e:
            return []

    def __extract_assistant_response(self, result: str) -> str:
        if "<|im_start|>assistant" in result:
            assistant_part = result.split("<|im_start|>assistant")[-1]
        else:
            assistant_part = result
        return assistant_part


class PlanExecutor:
    def follow_plan(self, plan: str) -> List[Dict[str, ReWOO]]:
        graph = StateGraph(ReWOO)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)
        
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self._route)
        graph.add_edge(START, "tool")
        app = graph.compile()
        result = []
        plan_steps = self.parse_plan(plan)
        status = {"plan_string": plan, "steps": plan_steps}
        for s in app.stream(status):
            result.append(s)
        return result
    
    def parse_plan(self, plan: str) -> List[tuple]:
        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, plan)
        return matches
    
    def _get_current_task(self, state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO):
        """Worker node that executes the tools of a given plan."""
        _step = self._get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        _results = (state["results"] or {}) if "results" in state else {}
        
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        
        if tool == "Calculator":
            try:
                expr = tool_input
                for k, v in _results.items():
                    # Remove '#' and ensure variables are strings or numbers
                    expr = expr.replace(k, str(v))
                    expr = expr.replace(k.lstrip('#'), str(v))
                
                expr = expr.replace("\"", "").replace("'", "")  # Remove quotes if any
                # Parse and evaluate safely
                evaluated = sympify(expr).evalf()
                result = str(evaluated)
            except SympifyError as e:
                result = f"SymPy Error: {e}"
            except Exception as e:
                result = f"Error: {e}"
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}

    def solve(self, state: ReWOO):
        try:
            last_value = float(list(state["results"].values())[-1])
            return {"result": last_value}
        except (ValueError, IndexError):
            result = "Error: Unable to retrieve the final result from the steps."
            return {"result": result, "message": []}
    
    def _route(self, state):
        _step = self._get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"