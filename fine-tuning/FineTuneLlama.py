import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from typing import List
from sklearn.model_selection import train_test_split

class FineTuneLlama:
    def __init__(self, model_name: str, file_path: str = "data/training_data.json", output_dir: str = "../training/models", num_epochs: int = 3):
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
        print('All attributes initialized...')

        # Prepare model, tokenizer, dataset, and training arguments
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print('Tokenizer initialized...')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.config.use_cache = False
        print('Model initialized...')

        self.train_dataset, self.val_dataset = self.prepare_dataset(self.file_path)
        print('Dataset loaded...')

        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=50,               # Evaluate every 50 steps
            learning_rate=5e-5,          # Increased learning rate for testing
            per_device_train_batch_size=4,  # Increased batch size if GPU memory allows
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            logging_dir=str(base_dir / 'logs'),
            logging_steps=10,
            fp16=True,
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",  # Define a metric if using custom metrics
            greater_is_better=True,
        )
        print('Training arguments configured...')

        # Initialize the Trainer with early stopping
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 evaluations
        )
        print('Trainer initialized...')

    def load_data(self, file_path: Path) -> List[dict]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def prepare_dataset(self, file_path: Path):
        train_data = self.load_data(file_path)

        # Split the data into training and validation sets
        train, val = train_test_split(train_data, test_size=0.1)
        train_dataset = Dataset.from_dict({"input": [entry["input"] for entry in train],
                                           "output": [entry["output"] for entry in train]})
        val_dataset = Dataset.from_dict({"input": [entry["input"] for entry in val],
                                         "output": [entry["output"] for entry in val]})

        def tokenize_function(examples):
            max_length = 350
            input_encodings = self.tokenizer(
                examples["input"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
            output_encodings = self.tokenizer(
                examples["output"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
            labels = output_encodings['input_ids']

            # Replace padding token ids in labels by -100 so that they are ignored in loss computation
            labels[labels == self.tokenizer.pad_token_id] = -100

            results =  {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': labels
            }

            print(results)
            return results;

        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        return train_dataset, val_dataset

    def start_training(self):
        print("Starting training...")
        train_result = self.trainer.train()
        print("Training complete!")

        # Evaluate on validation data
        eval_results = self.trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

        # Save the model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")


if __name__ == "__main__":
    # Free up unused GPU memory
    torch.cuda.empty_cache()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "./fine_tuned_model"

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    file_path = "data/training_data_1.json"
    output_dir = "../training/models/fine_tuned_model"

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    fine_tune_model = FineTuneLlama(model_name, file_path, output_dir)
    fine_tune_model.start_training()
