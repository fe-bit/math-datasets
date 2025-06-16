from .llm import LLM
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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
    def __init__(self, model_name: str, dtype: torch.dtype = torch.float32, quantization: bool = False):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map_strategy,
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=TransformerLLM.get_quantization_config() if quantization else None,
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

        answer = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Clean up intermediate tensors and flush CUDA cache
        del inputs, outputs, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Flush CUDA cache
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        
        return answer

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def __del__(self):
        """Clean up GPU memory when object is destroyed."""
        if hasattr(self, '_model') and torch.cuda.is_available():
            del self._model
            torch.cuda.empty_cache()

    @property
    def pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self._tokenizer,
        )
    
    @classmethod
    def get_quantization_config(cls) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="uint8",
        )