from math_datasets.datasets.base_dataset import Dataset


def get_compute_metrics(tokenizer):

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct = sum(
            Dataset.extract_answer(p) == Dataset.extract_answer(l)
            for p, l in zip(decoded_preds, decoded_labels)
        )
        final_accuracy = correct / len(decoded_preds)

        return {"final_answer_accuracy": final_accuracy}

    return compute_metrics
