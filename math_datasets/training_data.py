import pandas as pd
from .datasets.base_dataset import Dataset
from .evaluator import evaluate_detail
from typing import Callable


def get_training_data(model_name:str, dataset: Dataset, save_dir:str, label_function: Callable=None) -> pd.DataFrame:
    df = evaluate_detail(model_name, dataset, save_dir, use_transformated_answers=False)
    df_correct = df[df["is_correct"] == True].copy()
    if label_function is not None:
        df_correct["output"] = df_correct.apply(lambda row: label_function(row), axis=1)
    else:
        df_correct["output"] = df_correct["response"]

    df_correct["input"] = df_correct[dataset.prompt]
    df_correct = df_correct[["input", "output"]]
    return df_correct.reset_index(drop=True)