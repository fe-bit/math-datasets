from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from .generator import Generate
from langchain.schema import messages_to_dict
from typing import Any


@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression (e.g., '2 + 3 * 4')."""
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {str(e)}"
    

class ReactGenerate(Generate):
    def __init__(self, model: ChatOllama):
        self.model = model
        self.agent = create_react_agent(
            model=self.model,
            tools=[calculator],
            prompt="You are a math teacher solving math word problems and can perform calculations using a calculator tool."
        )

    def generate(self, prompt: str, entry: dict[str, str] = {}) -> str:
        try: 
            response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            entry["model_history"] = messages_to_dict(response["messages"])
            return response["messages"][-1].content
        except Exception as e:
            entry["model_history"] = str(e)
            return "Error occured."
        
    @classmethod
    def add_metrics(cls, entry: dict[str, Any]) -> dict[str, Any]:
        if entry["model_history"] == "Error occured.":
            entry["input_tokens"] = None
            entry["output_tokens"] = None
            entry["total_tokens"] = None
            entry["format_correct"] = False
        else:
            input_tokens, output_tokens, total_tokens = 0, 0, 0
            for i in entry["model_history"]:
                usage_metadata = i["data"].get("usage_metadata")
                if usage_metadata is not None:
                    input_tokens += usage_metadata["input_tokens"]
                    output_tokens += usage_metadata["output_tokens"]
                    total_tokens += usage_metadata["total_tokens"]
            
            entry["input_tokens"] = input_tokens
            entry["output_tokens"] = output_tokens
            entry["total_tokens"] = total_tokens
            entry["format_correct"] = True
            entry["reasoning"] = None
        return entry