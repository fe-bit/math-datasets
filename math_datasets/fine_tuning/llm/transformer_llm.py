from .llm import LLM
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft import PeftModel
import os


if torch.cuda.is_available():
    print("CUDA is available! Using GPU.")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
else:
    print("CUDA is not available. Using CPU.")
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Default to 1 if not in Slurm
    torch.set_num_threads(num_threads)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")


class TransformerLLM(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

    @classmethod
    def from_trained(cls, checkpoint_path: str, model_name: str):
        model = TransformerLLM(model_name=model_name)
        model._model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model._model.to(device)
        return model

    def apply_lora(self, lora_config: LoraConfig):
        self._model = get_peft_model(self._model, lora_config) # type: ignore

    def merge_with_peft(self, checkpoint_path: str):
        self._model = PeftModel.from_pretrained(self._model, checkpoint_path)
        self._model = self._model.merge_and_unload() # type: ignore

    def generate(self, text, max_new_tokens=200, temperature=0.7, top_p: float = 0.9, top_k: int = 50) -> str:
        text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0,
            top_k=top_k if temperature is not None else None,
            top_p=top_p if temperature is not None else None,
            
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

        # Decode only the new tokens (after the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer