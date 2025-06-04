# from datasets import load_dataset
# from ..base_dataset import Dataset
# import os


# class ASDiv(Dataset):
#     prompt = None
#     answer = "answer"
#     name = "ASDiv"
#     examples = None
#     priority = 3
#     ds = None
    
#     @classmethod
#     def get_dataset(cls):
#         if cls.ds is None:
#             cls.ds = load_dataset("json", data_files=os.path.join(os.path.abspath(os.path.dirname(__file__)), "ASDiv.jsonl"), split="train")
#             cls.ds = cls.ds.train_test_split(test_size=0.1, seed=42)
#         return cls.ds
    
#     @classmethod
#     def get_input_text(cls, example):
#         return example["body"] + " " + example["question"]
    
#     @classmethod
#     def get_float_answer(cls, example):
#         return cls.extract_answer(example["answer"])
    
#     @classmethod
#     def is_answer_correct(cls, entry, use_transformated_answers:bool=True) -> bool:
#         if use_transformated_answers:
#             answer = entry["extracted_response"]
#         else:
#             answer = entry["response"]
#         predicted = cls.extract_answer(answer)
#         if predicted is None:
#             return False
#         if predicted != cls.get_float_answer(entry):
#             return False
#         return True
    
#     @classmethod
#     def get_example_for(cls, example):
#         if cls.examples is None:
#             cls.load_examples()
#         return cls.examples[example["solution_type"]]

#     @classmethod
#     def load_examples(cls, num_examples_per_category=5):
#         dataset = cls.get_dataset()
#         unique_solution_types = len(set([i["solution_type"] for i in dataset["train"]]))
#         examples = {}
#         for i in dataset["train"]:
#             if i["solution_type"] in examples:
#                 if len(examples[i["solution_type"]]) < num_examples_per_category:
#                     examples[i["solution_type"]].append(f"""Question: {i["question"]}\Answer: {i['formula']}""")
#             else:
#                 examples[i["solution_type"]] = [f"""Question: {i["question"]}\Answer: {i['formula']}"""]
            
#             if all(len(examples[i]) >= num_examples_per_category for i in examples) and len(examples) == unique_solution_types:
#                 break
#         for i in examples:
#             examples[i] = "\n".join(examples[i])

#         cls.examples = examples
