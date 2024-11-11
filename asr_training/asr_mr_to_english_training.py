import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperTokenizer
from huggingface_hub import login
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
import torch
import evaluate

# Define the model name at the top-level
MODEL_NAME = "./whisper-small-mr-en-finetuned"

# Hugging Face login using your token (set this in the environment variable 'HUGGINGFACE_TOKEN')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

# Load and process custom dataset from CSV
def load_custom_dataset(csv_file):
    """
    Loads a custom dataset from CSV and prepares it for preprocessing.
    :param csv_file: Path to the CSV containing Marathi and English sentence pairs
    :return: Loaded dataset in a DatasetDict format
    """
    print("Loading custom dataset...")
    df = pd.read_csv(csv_file)
    
    # Prepare Dataset from CSV
    dataset = Dataset.from_pandas(df)
    dataset = DatasetDict({"train": dataset, "test": dataset})  # Use the same data for both train and test for now
    print("Custom dataset loaded.")
    return dataset

# Preprocess dataset function
def preprocess_data(dataset, processor, tokenizer, target_audio_length=3000):
    """
    Pre-processes the dataset by converting audio and text into input features.
    :param dataset: The loaded dataset
    :param processor: WhisperProcessor to process audio and text
    :param tokenizer: WhisperTokenizer to process text
    :param target_audio_length: The target length for audio features (mel spectrogram)
    :return: Preprocessed dataset
    """
    print("Preprocessing data...")

    def prepare_dataset(batch):
        print(batch.keys())
        # Load audio using librosa (this gives you a numpy array and sample rate)
        audio_path = os.path.join("/Users/pankajagrawal/desktop/projects/Paani/ai-voice-assistant/asr_training/dataset/mr-to-english", batch['audio_file'])
        print(f"Audio path: {audio_path}")
        
        # Ensure that the audio file exists before proceeding
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load the audio file and get the sampling rate
        audio_array, original_sampling_rate = librosa.load(audio_path, sr=None)  # Load with original sampling rate
        
        # Resample audio to 16000 Hz if needed
        if original_sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=original_sampling_rate, target_sr=16000)
            print(f"Resampled audio from {original_sampling_rate} Hz to 16000 Hz.")
        
        # Use the WhisperProcessor to process the audio and automatically pad or truncate
        audio_input = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        print(f"Audio input shape after padding: {audio_input['input_features'].shape}")
        
        # Tokenize English text (target language)
        english_target = tokenizer(batch["english_translation"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        batch["input_features"] = audio_input["input_features"]
        batch["labels"] = english_target["input_ids"].squeeze()  # Ensure labels are in the correct format
        return batch


    dataset = dataset.map(prepare_dataset, num_proc=3)  # Reduce num_proc to avoid too many processes
    print("Data preprocessed.")
    return dataset

# Create the Whisper model
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.language = "marathi"
    model.generation_config.task = "translate"  # Set task to translation
    model.config.gradient_checkpointing = True
    model.generation_config.forced_decoder_ids = None
    return model

# Data Collator for speech-to-text with padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Padding audio features using processor
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # Pad audio inputs using the processor's feature extractor
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)
        
        # Padding target labels (text)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        
        # Mask labels (where padding occurs)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Handle the case where the first label is the decoder start token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Compute metrics for evaluation (BLEU score for translation)
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = pred.tokenizer.pad_token_id
    pred_str = pred.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = pred.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # BLEU score for translation tasks
    bleu = evaluate.load("bleu")
    score = bleu.compute(predictions=pred_str, references=label_str)
    return {"bleu": score["bleu"]}

# Set up the trainer with all necessary arguments and configurations
def setup_trainer(dataset, model, processor):
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
        metric_for_best_model="bleu",  # Use BLEU as the evaluation metric
        greater_is_better=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    return trainer

# Main function to train and save the model
def train_and_save_model(csv_file):
    # Load custom dataset
    dataset = load_custom_dataset(csv_file)

    # Feature extractor and tokenizer
    feature_extractor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Marathi", task="translate")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="translate")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Marathi", task="translate")

    # Preprocess data
    dataset = preprocess_data(dataset, processor, tokenizer)

    # Load model
    model = load_model()

    # Set up trainer
    trainer = setup_trainer(dataset, model, processor)

    # Start training
    trainer.train()

    # Save model
    model.save_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_NAME)

    print("Model trained and saved successfully!")
    return model


# Push model to Hugging Face Hub (optional)
def upload_model_to_huggingface(model):
    model.push_to_hub(MODEL_NAME)
    print(f"Model uploaded to Hugging Face under the name: {MODEL_NAME}")


if __name__ == "__main__":
    # Provide the path to your CSV dataset
    csv_file = "./dataset/mr-to-english/mr-to-english.csv"
    train_and_save_model(csv_file)
    # Optionally, upload the model to Hugging Face
    # upload_model_to_huggingface(model)
