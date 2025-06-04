from transformers import Trainer, TrainingArguments
from peft import LoraConfig
from math_datasets.datasets import Dataset
from .llm import LLM


class CustomTrainer:
    def apply_lora(self, llm: LLM):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", # "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type="CAUSAL_LM",
            bias="lora_only"
        )
        llm.apply_lora(lora_config)
        print("LoRA applied.")

    def train(self, llm: LLM, output_dir: str, dataset: Dataset, resume_from_checkpoint: bool = False, training_args: TrainingArguments = None, apply_lora: bool = True):
        tokenized_dataset = dataset.load_and_tokenize_dataset(llm.tokenizer)

        if apply_lora:
            self.apply_lora(llm)

        if training_args is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,  # More accumulation for better effective batch size
                num_train_epochs=50,             # More epochs helps for GSM8K
                learning_rate=5e-5,             # Slightly higher; LoRA is robust to this
                warmup_steps=20,               # Helps stabilize training
                weight_decay=0.01,              # Adds regularization
                max_grad_norm=1.0,              # Prevent exploding gradients
                lr_scheduler_type="cosine",     # Better than constant
                logging_steps=100,
                save_steps=25,
                save_total_limit=2,
                eval_strategy="epoch",
                bf16=True,
                
                logging_dir=f"./logs/{llm.model_name}",
                run_name=llm.model_name,
                report_to="none",
            )
        
        with open(f"{output_dir}/training_args.json", "w") as f:
            f.write(training_args.to_json_string())

        # === Trainer Setup ===
        trainer = Trainer(
            model=llm.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir)
        llm.tokenizer.save_pretrained(output_dir)
