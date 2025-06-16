from datasets import load_dataset # type: ignore
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
    def get_input_text(cls, example):
        return example["question_concat"]
    
    @classmethod
    def get_output_text(cls, example) -> str:  
        return example["Answer"]

    
    @classmethod
    def get_float_answer(cls, example):
        return float(example["Answer"])
    
    @classmethod
    def is_answer_correct(cls, entry, use_transformated_answers:bool=False) -> bool:
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
    def format_input_evaluate(cls, example, prompt_prefix="") -> str:
        return (
            "<|im_start|>user\n" + prompt_prefix + cls.get_input_text(example) + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    @classmethod
    def load_and_tokenize_dataset(cls, tokenizer): # type: ignore
        def format_and_tokenize(example):
            prompt = (
                "<|im_start|>user\n" + cls.get_input_text(example) + "<|im_end|>\n"
                "<|im_start|>assistant\n" + cls.get_output_text(example) + "<|im_end|>"
            )
            model_inputs = tokenizer(
                prompt,
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        dataset = cls.get_dataset()
        print("Dataset loaded.")
        tokenized_dataset = dataset.map(format_and_tokenize, batched=False)
        tokenized_dataset.set_format("torch") # type: ignore
        return tokenized_dataset