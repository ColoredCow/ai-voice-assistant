import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Optional


class FineTuneLlama:
    def __init__(self, model_name: str, file_path: str, output_dir: str = "./results", num_epochs: int = 3):
        """
        Initializes the FineTuneLlama class.

        Args:
            model_name (str): The name of the pre-trained model to fine-tune.
            file_path (str): Path to the training data file (text file).
            output_dir (str): Directory to save the fine-tuned model and logs.
            num_epochs (int): Number of training epochs.
        """
        self.model_name = model_name
        self.file_path = file_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Prepare dataset
        self.train_dataset = self.prepare_dataset(file_path)

        # Prepare training arguments
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,               # Output directory for model checkpoints
            evaluation_strategy="epoch",              # Evaluate after each epoch
            learning_rate=2e-5,                       # Learning rate
            per_device_train_batch_size=8,            # Batch size for training
            per_device_eval_batch_size=8,             # Batch size for evaluation
            num_train_epochs=self.num_epochs,         # Number of training epochs
            weight_decay=0.01,                        # Weight decay for regularization
            logging_dir='./logs',                     # Directory for logging
            logging_steps=10,                         # Log every 10 steps
        )

        # Prepare Trainer
        self.trainer = Trainer(
            model=self.model,                         # Pre-trained model
            args=self.training_args,                  # Training arguments
            train_dataset=self.train_dataset,         # Training dataset
            tokenizer=self.tokenizer                  # Tokenizer for preprocessing inputs
        )

    def load_data(self, file_path: str) -> List[str]:
        """
        Loads the training data from a text file.

        Args:
            file_path (str): Path to the text file containing training data.

        Returns:
            List[str]: List of training examples.
        """
        with open(file_path, 'r') as file:
            return file.readlines()

    def prepare_dataset(self, file_path: str) -> Dataset:
        """
        Prepares the dataset by loading and tokenizing the JSON data file.

        Args:
            file_path (str): Path to the JSON file containing training data.

        Returns:
            Dataset: Tokenized Hugging Face Dataset.
        """
        train_data = self.load_data(file_path)

        # Create a dataset using Hugging Face Dataset.from_dict
        train_dataset = Dataset.from_dict(
            {"input": train_data["input"], "output": train_data["output"]})

        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize both input and output fields
            input_encodings = self.tokenizer(
                examples["input"], padding=True, truncation=True, return_tensors="pt")
            output_encodings = self.tokenizer(
                examples["output"], padding=True, truncation=True, return_tensors="pt")
            return {
                'input_ids': input_encodings['input_ids'],
                'labels': output_encodings['input_ids']
            }

        # Apply the tokenization to the dataset
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        return train_dataset

    def start_training(self):
        """
        Starts the fine-tuning process.
        """
        print("Starting training...")
        self.trainer.train()
        print("Training complete!")

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")


if __name__ == "__main__":
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    file_path = "training_data.json"
    output_dir = "./fine_tuned_model"

    fine_tune_model = FineTuneLlama(model_name, file_path, output_dir)

    fine_tune_model.start_training()
