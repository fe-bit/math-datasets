from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig
from math_datasets.datasets import Dataset
from .llm import LLM
from .llm.utils import get_latest_checkpoint_dir
import os
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from math_datasets.fine_tuning.metrics.compute_metrics import get_compute_metrics


class CustomTrainer:
    def train(self, llm: LLM, output_dir: str, dataset: Dataset, resume_from_checkpoint: bool, training_args: TrainingArguments, lora_config: LoraConfig|None = None):
        tokenized_dataset = dataset.load_and_tokenize_dataset(llm.tokenizer)

        training_args = SFTConfig(packing=True, **training_args.to_dict())
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/training_args.json", "w") as f:
            f.write(training_args.to_json_string())

        trainer = SFTTrainer(
            model=llm.model,
            args=training_args,
            peft_config=lora_config, # is None if not using LoRA
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=get_compute_metrics(llm.tokenizer)
        )

        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=get_latest_checkpoint_dir(output_dir=output_dir))
        else:
            trainer.train()
            
        trainer.save_model(output_dir)
        llm.tokenizer.save_pretrained(output_dir)
