from math_datasets.datasets.base_dataset import Dataset
import numpy as np


def get_compute_metrics(tokenizer):
    def is_correct_answer(prediction: float|None, label: float|None) -> bool:
        if prediction is None:
            return False
        elif label is None:
            raise ValueError("Label cannot be None")
        return prediction == label
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # Convert logits to token IDs if needed
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Replace -100 with pad_token_id in labels
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct = 0
        for p, l in zip(decoded_preds, decoded_labels):
            pred_answer = Dataset.extract_answer(p)
            label_answer = Dataset.extract_answer(l)
            print(f"RAW PRED: {p}\nRAW LABEL: {l}")
            print(f"PRED: {pred_answer} — LABEL: {label_answer}")

            correct += is_correct_answer(pred_answer, label_answer)

        final_accuracy = correct / len(predictions)

        return {"math_accuracy": final_accuracy}

    return compute_metrics
