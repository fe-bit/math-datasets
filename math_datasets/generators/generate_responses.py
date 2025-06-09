import json
from tqdm import tqdm
import os
from math_datasets.datasets.base_dataset import Dataset
from .generator import Generate
from typing import Literal


def get_output_file(save_dir, model_name: str, dataset: Dataset) -> str:
    return f"{save_dir}/evaluations/{dataset.name}/{model_name}.jsonl"

def generate_responses(dataset: Dataset, model_name: str, generator:Generate, save_dir:str, first_n:int|None=None, dataset_split: Literal["test", "train"]="test", overwrite: bool=False):
    ds = dataset.get_dataset()
    output_path = get_output_file(save_dir, model_name, dataset)
    
    if first_n is not None:
        if first_n > len(ds[dataset_split]):
            print(f"Limit {first_n} is greater than the number of samples {len(ds['test'])}. Use the full dataset.")
        else:
            print("Limiting to first", first_n, "samples.")
            ds[dataset_split] = ds[dataset_split].select(range(first_n))
    if overwrite and os.path.exists(output_path):
        print(f"🗑️ {model_name}: Overwriting existing evaluation results for {dataset.name}.")
        os.remove(output_path)
    try:
        with open(output_path, "r") as f:
            start_idx = len([json.loads(line) for line in f])
    except FileNotFoundError:
        start_idx = 0

    if start_idx >= len(ds[dataset_split]):
        print(f"✅ {model_name}: Already evaluated all {dataset.name} test samples. ({start_idx}/{len(ds['test'])} samples evaluated)")
        return
    elif start_idx == 0:
        print(f"🚀 {model_name}: Starting evaluation on {dataset.name} test set.")
    else:
        print(f"🔄 {model_name}: Resuming evaluation on {dataset.name} test set from index {start_idx}.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a") as f:
        for i in tqdm(ds[dataset_split].select(range(start_idx, len(ds[dataset_split])))):
            prompt = dataset.get_input_text(i)
            response = generator.generate(prompt, i)
            i["response"] = response
            f.write(json.dumps(i) + "\n")



    