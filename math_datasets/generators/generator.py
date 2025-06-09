import ollama
from abc import abstractmethod
from google import genai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel
from math_datasets.fine_tuning.llm.transformer_llm import TransformerLLM


class Generate:
    @abstractmethod
    def generate(self, prompt: str,  entry: dict[str, str]={}) -> str:
        """
        Generate a response using the Ollama model.
        """

class OllamaGenerate(Generate):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate(self, prompt:str,  entry: dict[str, str]={}) -> str:
        return ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "num_predict": 512,
                "stop": ["<|eot|>"],
            }
        )["response"]


class GeminiGenerate(Generate):
    def __init__(self, model_name="gemini-2.0-flash", wait_frequency=5, num_retries: int|None=None):
        self.model_name = model_name
        self.client = genai.Client()
        self.wait_frequency = wait_frequency
        self.num_retries = num_retries
    
    def generate(self, prompt:str,  entry: dict[str, str]={}) -> str:
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
