import os
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pandas as pd
import evaluate

# Define the model name at the top-level
MODEL_NAME = "./whisper-small-mr-en-translation"

# Function to load the CSV dataset
def load_translation_dataset():
    """
    Loads the Marathi-English translation dataset from a CSV file and prepares it for training.
    :return: Loaded dataset
    """
    print("Loading translation dataset...")
    # Load the CSV file
    df = pd.read_csv("./data/marathi_to_english.csv")
    
    # Prepare the dataset
    audio_files = [os.path.join("./data/audio", f) for f in df["audio_file"]]
    marathi_sentences = df["marathi_sentence"].tolist()
    english_translations = df["english_translation"].tolist()

    dataset = Dataset.from_dict({
        "audio": audio_files,
        "marathi_sentence": marathi_sentences,
        "english_translation": english_translations
    })

    # Cast the audio column to Audio format with 16kHz sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("Dataset loaded.")
    return dataset

# Preprocess the data for translation
def preprocess_translation_data(dataset, feature_extractor, tokenizer):
    """
    Pre processes the dataset for translation by converting audio and text into input features.
    :param dataset: The loaded dataset
    :param feature_extractor: WhisperFeatureExtractor to process audio
    :param tokenizer: WhisperTokenizer to process text
    :return: Preprocessed dataset
    """
    print("Preprocessing data...")
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["english_translation"]).input_ids
        return batch

    dataset = dataset.map(lambda batch: prepare_dataset(batch), remove_columns=["marathi_sentence", "english_translation"], num_proc=4)
    print("Data preprocessed.")
    return dataset

# Create the Whisper model for translation
def load_model_for_translation():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.language = "marathi"
    model.generation_config.task = "translate"
    model.config.gradient_checkpointing = True
    # model.generation_config.forced_decoder_ids = None
    return model

# Data Collator for translation task
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Compute metrics for evaluation (using BLEU or other translation metrics)
def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Use BLEU for translation evaluation
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=pred_str, references=label_str)
    return bleu_score


# Set up the trainer with all necessary arguments and configurations
def setup_trainer_for_translation(dataset, model, processor, tokenizer):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_NAME,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=100,
        gradient_checkpointing=True,
        fp16=False,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,  # You can split your dataset into train and eval if needed
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=processor.feature_extractor,
    )
    return trainer

# Main function to train and save the model
def train_and_save_translation_model():
    # Load data
    dataset = load_translation_dataset()

    # Feature extractor and tokenizer for translation
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="translate")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Marathi", task="translate")

    # Preprocess data
    dataset = preprocess_translation_data(dataset, feature_extractor, tokenizer)

    # Load model
    model = load_model_for_translation()

    # Set up trainer
    trainer = setup_trainer_for_translation(dataset, model, processor, tokenizer)

    # Start training
    trainer.train()

    # Save model
    model.save_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_NAME)
    feature_extractor.save_pretrained(MODEL_NAME)

    print("Model trained and saved successfully!")
    return model

# Push model to Hugging Face Hub (optional)
def upload_model_to_huggingface(model):
    model.push_to_hub(MODEL_NAME)
    print(f"Model uploaded to Hugging Face under the name: {MODEL_NAME}")


if __name__ == "__main__":
    train_and_save_translation_model()
    # Optionally, upload the model to Hugging Face
    # upload_model_to_huggingface(model)
