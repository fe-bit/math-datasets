import ollama
from abc import abstractmethod
from google import genai
import time
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM
from .rewoo import ReWOOModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_ollama import ChatOllama
from typing import Any


class Generate:
    @abstractmethod
    def generate(self, prompt: str,  entry: dict[str, str]={}) -> str:
        """
        Generate a response using the Ollama model.
        """
    
    @classmethod
    def add_metrics(cls, entry: dict[str, Any]) -> dict[str, str]:
        return entry

class OllamaGenerate(Generate):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate(self, prompt:str,  entry: dict[str, str]={}) -> str:
        resp = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "num_predict": 512,
                "stop": ["<|eot|>"],
            }
        )
        entry["usage_metadata"] = {
            "input_tokens": resp["prompt_eval_count"],
            "output_tokens": resp["eval_count"],
            "total_tokens": resp["prompt_eval_count"] + resp["eval_count"]
        }
        return resp["response"]
    
    @classmethod
    def add_metrics(cls, entry: dict[str, Any]) -> dict[str, str]:
        try:
            entry["input_tokens"] = entry["usage_metadata"]["input_tokens"]
            entry["output_tokens"] = entry["usage_metadata"]["output_tokens"]
            entry["total_tokens"] = entry["usage_metadata"]["total_tokens"]
        except:
            entry["input_tokens"] = None
            entry["output_tokens"] = None
            entry["total_tokens"] = None
        return entry


class GeminiGenerate(Generate):
    def __init__(self, model_name="gemini-2.0-flash", wait_frequency=5, num_retries: int|None=None):
        self.model_name = model_name
        self.client = genai.Client()
        self.wait_frequency = wait_frequency
        self.num_retries = num_retries
    
    def generate(self, prompt:str,  entry: dict[str, Any]={}) -> str:
        num_retries = self.num_retries
        while num_retries is None or num_retries > 0:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                )
                time.sleep(self.wait_frequency)
                entry["usage_metadata"] = response.usage_metadata.model_dump()
                return response.text
            except:
                if num_retries is not None:
                    num_retries -= 1
                time.sleep(self.wait_frequency*10)

        return "Error occured."
    
    @classmethod
    def add_metrics(cls, entry: dict[str, Any]) -> dict[str, str]:
        entry["input_tokens"] = entry["usage_metadata"]["prompt_token_count"]
        entry["total_token_count"] = entry["usage_metadata"]["total_token_count"]
        entry["output_tokens"] = entry["total_token_count"] - entry["input_tokens"]
        return entry

class TransformersGenerate(Generate):
    def __init__(self, model: TransformerLLM, temperature=None, top_k=None, top_p=None, max_new_tokens=512):
        self.model = model

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
    def generate(self, prompt: str, entry: dict[str, str] = {}) -> str:
        return self.model.generate(
            text=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            entry=entry
        )
    
    @classmethod
    def add_metrics(cls, entry: dict[str, str]) -> dict[str, Any]:
        try:
            entry["input_tokens"] = entry["usage_metadata"]["input_tokens"]
            entry["output_tokens"] = entry["usage_metadata"]["output_tokens"]
            entry["total_tokens"] = entry["usage_metadata"]["total_tokens"]
        except:
            entry["input_tokens"] = None
            entry["output_tokens"] = None
            entry["total_tokens"] = None
        return entry


class ReWOOGenerate(Generate):
    def __init__(self, rewoo_model: ReWOOModel, sleep_time: int=5, retries: int=1):
        self.rewoo_model = rewoo_model
        self.sleep_time = sleep_time
        self.retries = retries

    @classmethod
    def init_gemini(cls, model_name: str = "gemini-2.0-flash", sleep_time: int = 5):
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        rewoo_model = ReWOOModel(model=model, sleep_time=5)
        return ReWOOGenerate(rewoo_model=rewoo_model, sleep_time=sleep_time)
    
    @classmethod
    def init_ollama(cls, model_name: str, with_examples: bool = True):
        model = ChatOllama(model=model_name, temperature=0, num_predict=1024)
        rewoo_model = ReWOOModel(model=model, sleep_time=0, with_examples=with_examples)
        return ReWOOGenerate(rewoo_model=rewoo_model, retries=0, sleep_time=0)
    
    @classmethod
    def init_transformer_llm(cls, llm: TransformerLLM, sleep_time: int = 0, with_examples: bool = True):
        llm.model.eval()  # Ensure the model is in evaluation mode
        llm_pipeline = HuggingFacePipeline(pipeline=llm.pipeline, model_kwargs={"temperature": 0.0, "max_new_tokens": 512})
        chat_model = ChatHuggingFace(llm=llm_pipeline)
        rewoo_model = ReWOOModel(model=chat_model, sleep_time=sleep_time, with_examples=with_examples)
        return ReWOOGenerate(rewoo_model=rewoo_model, sleep_time=sleep_time, retries=0)
    
    @classmethod
    def get_chat_huggingface(cls, llm: TransformerLLM, sleep_time: int = 0):
        llm.model.eval()  # Ensure the model is in evaluation mode
        llm_pipeline = HuggingFacePipeline(pipeline=llm.pipeline, model_kwargs={"temperature": 0.0, "max_new_tokens": 512})
        chat_model = ChatHuggingFace(llm=llm_pipeline)
        return chat_model

    def generate(self, prompt, entry: dict[str, str]={}) -> str:
        counter = 0
        while True:
            try:
                time.sleep(self.sleep_time)
                resp = self.rewoo_model(prompt)
                entry["model_history"] = resp
                return resp[-1]["solve"]["result"]
            except Exception as e:
                counter += 1
                if counter > self.retries:
                    entry["model_history"] = "Error occured."
                    return "Error occured."
                print(f"Error: {e}")
                print(f"Retrying in {2*self.sleep_time} seconds...")
                time.sleep(2*self.sleep_time)
    
    @classmethod
    def add_metrics(cls, entry: dict[str, Any]) -> dict[str, Any]:
        model_history = entry.get("model_history")
        if model_history:
            if model_history == "Error occured.":
                entry["format_correct"] = False
                entry["input_tokens"] = None
                entry["output_tokens"] = None
                entry["total_tokens"] = None
            else:
                entry["format_correct"] = True
                usage_metadata = model_history[0]["plan"]["message"][0]["data"]["usage_metadata"]
                entry["input_tokens"] = usage_metadata["input_tokens"]
                entry["output_tokens"] = usage_metadata["output_tokens"]
                entry["total_tokens"] = usage_metadata["total_tokens"]
        return entry