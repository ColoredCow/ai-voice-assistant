import json
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Optional


class FineTuneLlama:
    def __init__(self, model_name: str, file_path: str = "data/training_data.json", output_dir: str = "./results", num_epochs: int = 3):
        """
        Initializes the FineTuneLlama class.

        Args:
            model_name (str): The name of the pre-trained model to fine-tune.
            file_path (str): Path to the training data file (relative to project).
            output_dir (str): Directory to save the fine-tuned model and logs.
            num_epochs (int): Number of training epochs.
        """
        # Set up paths
        base_dir = Path(__file__).resolve().parent
        self.file_path = base_dir / file_path
        self.output_dir = base_dir / output_dir

        # Initialize attributes
        self.model_name = model_name
        self.num_epochs = num_epochs

        # Prepare model, tokenizer, dataset, and training arguments
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.train_dataset = self.prepare_dataset(self.file_path)
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),  # Convert Path to string
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            logging_dir=str(base_dir / 'logs'),
            logging_steps=10,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer
        )

    def load_data(self, file_path: Path) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def prepare_dataset(self, file_path: Path) -> Dataset:
        train_data = self.load_data(file_path)
        train_dataset = Dataset.from_dict(
            {"input": [entry["input"] for entry in train_data],
             "output": [entry["output"] for entry in train_data]}
        )

        def tokenize_function(examples):
            input_encodings = self.tokenizer(
                examples["input"], padding=True, truncation=True, return_tensors="pt")
            output_encodings = self.tokenizer(
                examples["output"], padding=True, truncation=True, return_tensors="pt")
            return {'input_ids': input_encodings['input_ids'], 'labels': output_encodings['input_ids']}

        return train_dataset.map(tokenize_function, batched=True)

    def start_training(self):
        print("Starting training...")
        self.trainer.train()
        print("Training complete!")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")


if __name__ == "__main__":
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    file_path = "data/training_data.json"
    output_dir = "./fine_tuned_model"

    fine_tune_model = FineTuneLlama(model_name, file_path, output_dir)
    fine_tune_model.start_training()
