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
        print('all attributes initialized....')

        # Prepare model, tokenizer, dataset, and training arguments
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print('tokenizer initialized....')

        # # Load the training data
        # file_path = self.file_path
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     training_data = json.load(f)


        # # Calculate token lengths for inputs and outputs
        # input_lengths = [len(self.tokenizer.encode(entry["input"], truncation=False)) for entry in training_data]
        # output_lengths = [len(self.tokenizer.encode(entry["output"], truncation=False)) for entry in training_data]

        # # Summary statistics
        # input_length_summary = {
        #     "max_input_length": max(input_lengths),
        #     "average_input_length": sum(input_lengths) / len(input_lengths),
        #     "min_input_length": min(input_lengths)
        # }

        # output_length_summary = {
        #     "max_output_length": max(output_lengths),
        #     "average_output_length": sum(output_lengths) / len(output_lengths),
        #     "min_output_length": min(output_lengths)
        # }

        # print("Input length summary:", input_length_summary)
        # print("Output length summary:", output_length_summary)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        print('model initialized....')
        self.train_dataset = self.prepare_dataset(self.file_path)
        print('dataset loaded....')
        print(self.train_dataset)
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),  # Convert Path to string
            eval_strategy="no",
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            logging_dir=str(base_dir / 'logs'),
            logging_steps=10,
            fp16=True,
        )
        print('training args configured....')
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer
        )
        print('trainer initialized....')

    def load_data(self, file_path: Path) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def prepare_dataset(self, file_path: Path) -> Dataset:
        print('before train data.....')
        train_data = self.load_data(file_path)
        print('after train data.....')
        train_dataset = Dataset.from_dict(
            {"input": [entry["input"] for entry in train_data],
             "output": [entry["output"] for entry in train_data]}
        )
        print('after train dataset.....')

        def tokenize_function(examples):
            print('inside tokenize_function.....')
            max_length=350
            input_encodings = self.tokenizer(examples["input"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            output_encodings = self.tokenizer(examples["output"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            # Print shapes for debugging
            print("input_ids shape:", input_encodings['input_ids'].shape)
            print("labels shape:", output_encodings['input_ids'].shape)
            return {'input_ids': input_encodings['input_ids'], 'labels': output_encodings['input_ids']}

        print('before train dataset return.....')
        return train_dataset.map(tokenize_function, batched=True)

    def start_training(self):
        print("Starting training...")
        self.trainer.train()
        print("Training complete!")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")


if __name__ == "__main__":
    # Free up unused GPU memory
    torch.cuda.empty_cache()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    file_path = "data/training_data.json"
    output_dir = "./fine_tuned_model"

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    fine_tune_model = FineTuneLlama(model_name, file_path, output_dir)
    fine_tune_model.start_training()
