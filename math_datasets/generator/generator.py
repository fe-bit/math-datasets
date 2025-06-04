import ollama
from abc import abstractmethod
from google import genai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel


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
    def __init__(self, model_name="gemini-2.0-flash", wait_frequency=5):
        self.model_name = model_name
        self.client = genai.Client()
        self.wait_frequency = wait_frequency
    
    def generate(self, prompt:str,  entry: dict[str, str]={}) -> str:
        time.sleep(self.wait_frequency)
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        entry["usage_metadata"] = response.usage_metadata.model_dump()
        return response.text


class TransformersGenerate(Generate):
    def __init__(self, model_name: str, temperature=None, top_k=None, top_p=None, max_new_tokens=512):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
    def generate(self, prompt: str, entry: dict[str, str] = {}) -> str:
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        input_token_count = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature is not None and self.temperature > 0,
                top_k=self.top_k,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
        generated_tokens = outputs[0][input_token_count:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        output_token_count = generated_tokens.shape[0]
        total_token_count = input_token_count + output_token_count
        entry["usage_metadata"] = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": total_token_count,
        }
        return response
    
    def merge_with_peft(self, checkpoint_path: str):
        self._model = PeftModel.from_pretrained(self._model, checkpoint_path)
        self._model = self._model.merge_and_unload()
