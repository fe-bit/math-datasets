from .llm import LLM
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft import PeftModel
import os
from trl import setup_chat_format
from typing import Any


if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device_map_strategy = "cuda"
elif torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
    device_map_strategy = "mps"
else:
    print("CUDA is not available. Using CPU.")
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Default to 1 if not in Slurm
    torch.set_num_threads(num_threads)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")
    device_map_strategy = "cpu"


class TransformerLLM(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer.eos_token = "<|im_end|>"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map_strategy,
            trust_remote_code=True,
        )
        # Store the actual device the model was loaded to
        if isinstance(self._model.device, torch.device):
            self.device = self._model.device
        else:
            print(f"Warning: Model loaded with a device_map that may distribute layers. Model device is: {self._model.device}")
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_trained(cls, checkpoint_path: str, model_name: str):
        # This method is fine, but also consider using device_map
        model = TransformerLLM(model_name=model_name)
        model._model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=model.device, # Use the same device as the class
            trust_remote_code=True # Add trust_remote_code if needed
        )
        return model

    def apply_lora(self, lora_config: LoraConfig):
        self._model = get_peft_model(self._model, lora_config) # type: ignore
        print(f"LoRA applied. Model type is now: {type(self._model)}") # Add this for debugging

    def merge_with_peft(self, checkpoint_path: str):
        self._model = PeftModel.from_pretrained(self._model, checkpoint_path)
        self._model = self._model.merge_and_unload() # type: ignore

    def generate(self, text, max_new_tokens=512, temperature=0, top_p: float = 0.9, top_k: int = 50, entry: dict[str, Any]={}) -> str:
        messages = [
            {"role": "user", "content": text},
        ]
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # <--- KEY CHANGE: Get the string representation
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0, # Only sample if temperature > 0
            top_k=top_k if temperature is not None else None,
            top_p=top_p if temperature is not None else None,
            
            pad_token_id=self._tokenizer.eos_token_id,
            # For Qwen and similar chat models, <|im_end|> is often the stop token.
            # Using convert_tokens_to_ids is robust here.
            eos_token_id=self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        
        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = generated_tokens.shape[0]
        total_token_count = input_token_count + output_token_count
        
        entry["usage_metadata"] = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": total_token_count,
        }
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer