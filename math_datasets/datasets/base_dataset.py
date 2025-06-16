import re
from abc import ABC, abstractmethod
from datasets.dataset_dict import DatasetDict


class Dataset(ABC):
    prompt = None
    answer = None
    name = None
    priority = None
    ds = None
    examples = None
    
    @classmethod
    @abstractmethod
    def get_dataset(cls) -> DatasetDict:
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    @abstractmethod
    def get_input_text(cls, example):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    @abstractmethod
    def get_output_text(cls, example) -> str:  
        pass
    
    @classmethod
    @abstractmethod
    def get_float_answer(cls, example):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def extract_answer(cls, text):
        if isinstance(text, float) or isinstance(text, int):
            return float(text)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
        return None
        
    @classmethod
    @abstractmethod
    def is_answer_correct(cls, entry: dict, use_transformated_answers:bool=False) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def clear_cache(cls):
        cls.ds = None
        cls.examples = None

    @classmethod
    @abstractmethod
    def format_input_evaluate(cls, example, prompt_prefix="") -> str:
        pass

    @classmethod
    @abstractmethod
    def load_and_tokenize_dataset(cls, tokenizer) -> DatasetDict:
        pass

