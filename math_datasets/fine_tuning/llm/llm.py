from abc import ABC, abstractmethod
from peft import LoraConfig


class LLM(ABC):
    model_name: str
    
    @property
    def model(self):
        pass

    @property
    def tokenizer(self):
        pass

    @abstractmethod
    def apply_lora(self, lora_config: LoraConfig):
        pass

    @abstractmethod
    def merge_with_peft(self, checkpoint_path: str):
        pass

    @abstractmethod
    def generate(self, text, max_new_tokens=200, temperature=0.7) -> str:
        pass
