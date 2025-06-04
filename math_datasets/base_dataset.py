import re


class Dataset:
    prompt = None
    answer = None
    name = None
    priority = None
    ds = None
    examples = None
    
    @classmethod
    def get_dataset(cls):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def get_prompt(cls, example):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def get_float_answer(cls, example):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def extract_answer(cls, text):
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            number = numbers[-1]
            if ":" in number:
                a, b = map(int, number.split(':'))
                number = a / b
            return float(number)
        else:
            return None
        
    @classmethod
    def is_answer_correct(cls, entry: dict, use_transformated_answers:bool=True) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def clear_cache(cls):
        cls.ds = None
        cls.examples = None

    @classmethod
    def get_example_for(cls, example):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def load_examples(cls, num_examples_per_category=5):
        raise NotImplementedError("Subclasses should implement this method.")