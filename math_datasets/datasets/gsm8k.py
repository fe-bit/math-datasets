from datasets import load_dataset # type: ignore
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
    def get_input_text(cls, example):
        return example["question"]
    
    @classmethod
    def get_output_text(cls, example):  
        return example["answer"]
    
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
    def format_input_evaluate(cls, example, prompt_prefix=""):
        return (
            "<|im_start|>user\n" + prompt_prefix + cls.get_input_text(example) + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    @classmethod
    def load_and_tokenize_dataset(cls, tokenizer): # type: ignore
        def format_and_tokenize(example):
            prompt = (
                "<|im_start|>user\n" + cls.get_input_text(example) + "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            output = cls.get_output_text(example) + "<|im_end|>"

            prompt_ids = tokenizer(prompt, truncation=True)["input_ids"]
            output_ids = tokenizer(output, truncation=True)["input_ids"]

            input_ids = prompt_ids + output_ids
            labels = [-100] * len(prompt_ids) + output_ids

            # Truncate to max length
            max_length = 2000
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        dataset = cls.get_dataset()
        print("Dataset loaded.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_dataset = dataset.map(format_and_tokenize, batched=False)
        tokenized_dataset.set_format("torch")
        return tokenized_dataset