from math_datasets.datasets.base_dataset import Dataset
from math_datasets.generators.generate_responses import get_output_file, Generate
import json
import numpy as np
import pandas as pd
from typing import Type


def evaluate_entry(entry, dataset: Dataset, use_transformated_answers:bool=False):
    return dataset.is_answer_correct(entry, use_transformated_answers)

def evaluate(model_name, datasets: list[Dataset], save_dir: str, use_transformated_answers=True, use_first_n:int|None=None):
    result = {
        "model": model_name
    }
    for dataset in datasets:
        output_path = get_output_file(save_dir, model_name, dataset)
        key = dataset.name
        try:
            with open(output_path, "r") as f:
                entries = [json.loads(line) for line in f]            
        except FileNotFoundError:
            print(f"❌ {model_name}: No output file found for {dataset.name}.")
            
            result[dataset.name] = np.nan
            continue
        if use_first_n is not None:
            entries = entries[:use_first_n]
        evaluated_entries = []
        for entry in entries:
            try:
                is_correct = evaluate_entry(entry, dataset, use_transformated_answers)
                evaluated_entries.append(is_correct)
            except Exception as e:
                print(f"Error evaluating entry in {dataset.name} for model {model_name}: {e}")
                pass
        if len(evaluated_entries) == 0:
            print(f"❌ {model_name}: {dataset.name} not transformed yet.")
            result[key] = np.nan
        elif len(evaluated_entries) != len(entries):
            print(f"🔄 {model_name}: {dataset.name} not entirely transformed yet.")
            result[key] = round(sum(evaluated_entries) / len(evaluated_entries) * 100, 2)
        else:
            result[key] = round(sum(evaluated_entries) / len(evaluated_entries) * 100, 2)
    
    return result

def evaluate_all(model_names: list[str], datasets: list[Dataset], save_dir: str, use_transformated_answers=False, use_first_n:int|None=None) -> pd.DataFrame:
    results = []
    for model_name in model_names:
        result = evaluate(model_name, datasets, save_dir, use_transformated_answers=use_transformated_answers, use_first_n=use_first_n)
        results.append(result)
    df = pd.DataFrame(results)
    return df


def evaluate_detail(model_name:str, dataset: Type[Dataset], save_dir:str, use_transformated_answers=False, additional_metrics: Type[Generate]|None=None) -> pd.DataFrame:
    """Evaluate a single model on a single dataset and return the results as a DataFrame."""
    output_path = get_output_file(save_dir, model_name, dataset)
    try:
        with open(output_path, "r") as f:
            entries = [json.loads(line) for line in f]            
    except FileNotFoundError:
        print(f"❌ {model_name}: No output file found for {dataset.name}.")
        return pd.DataFrame()
    
    evaluated_entries = []
    for entry in entries:
        try:
            is_correct = evaluate_entry(entry, dataset, use_transformated_answers)
            entry["is_correct"] = is_correct
            entry["question"] = dataset.get_input_text(entry)
            entry["target_answer"] = dataset.get_output_text(entry)
            if additional_metrics:
                entry = additional_metrics.add_metrics(entry)
            evaluated_entries.append(entry)
        except Exception as e:
            pass
    
    df = pd.DataFrame(evaluated_entries)
    return df