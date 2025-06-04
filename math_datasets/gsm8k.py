from datasets import load_dataset
from .base_dataset import Dataset


class GSM8K(Dataset):
    prompt = "question"
    answer = "answer"
    name = "GSM8K"
    priority = 1
    ds = None
    examples = None
    
    @classmethod    
    def get_dataset(cls):
        if cls.ds is None:
            cls.ds = load_dataset("openai/gsm8k", "main")
        return cls.ds

    @classmethod
    def get_prompt(cls, example):
        return example["question"]
    
    @classmethod
    def get_float_answer(cls, example):
        return float(example["answer"].split("####")[-1].strip().replace(",", ""))#.replace(",", ".").strip())
    
    @classmethod
    def is_answer_correct(cls, entry, use_transformated_answers:bool=True) -> bool:
        if use_transformated_answers:
            answer = entry["extracted_response"]
        else:
            answer = entry["response"]
        predicted = cls.extract_answer(answer)
        if predicted is None:
            return False
        if predicted != cls.get_float_answer(entry):
            return False
        return True
    
    @classmethod
    def get_example_for(cls, example):
        if cls.examples is None:
            cls.load_examples()
        return cls.examples
    
    @classmethod
    def load_examples(cls, num_examples_per_category=5):
        dataset = cls.get_dataset()
        examples = []
        for i in dataset["train"].select(range(num_examples_per_category)):
            examples.append(f"""Question: {i["question"]}\Answer: {i['answer']}""")
        examples = "\n".join(examples)
        cls.examples = examples