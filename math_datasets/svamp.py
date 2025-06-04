from datasets import load_dataset
from .base_dataset import Dataset


class SVAMP(Dataset):
    prompt = "question_concat"
    answer = "Answer"
    name = "SVAMP"
    priority = 2
    ds = None
    examples = None
    
    @classmethod
    def get_dataset(cls):
        if cls.ds is None:
            cls.ds = load_dataset("ChilleD/SVAMP")
        return cls.ds

    @classmethod
    def get_prompt(cls, example):
        return example["question_concat"]
    
    @classmethod
    def get_float_answer(cls, example):
        return float(example["Answer"])
    
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
        return cls.examples[example["Type"]]
    
    @classmethod
    def load_examples(cls, num_examples_per_category=5):
        dataset = cls.get_dataset()
        unique_solution_types = len(set([i["Type"] for i in dataset["train"]]))
        examples = {}
        for i in dataset["train"]:
            if i["Type"] in examples:
                if len(examples[i["Type"]]) < num_examples_per_category:
                    examples[i["Type"]].append(f"""Question: {i["question_concat"]}\Answer: {i['Equation']} = {i['Answer']}""")
            else:
                examples[i["Type"]] = [f"""Question: {i["question_concat"]}\Answer: {i['Equation']} = {i['Answer']}"""]
            
            if all(len(examples[i]) >= num_examples_per_category for i in examples) and len(examples) == unique_solution_types:
                break
        for i in examples:
            examples[i] = "\n".join(examples[i])
        cls.examples = examples