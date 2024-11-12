import os
from datasets import load_dataset, DatasetDict, Audio, Dataset
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperTokenizer
from huggingface_hub import login
from huggingface_hub import HfApi
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from dotenv import load_dotenv

load_dotenv()

# Define the model name at the top-level

current_path = os.path.abspath(__file__)
project_root = current_path[:current_path.index("ai-voice-assistant") + len("ai-voice-assistant")]

MODEL_NAME = os.path.join(project_root, "training", "models", "whisper-small-mr-finetuned-marathi-2")


# Hugging Face login using your token (set this in the environment variable 'HUGGINGFACE_TOKEN')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

def load_custom_dataset_function():
    """
    Loads the custom dataset and prepares it for preprocessing.
    :return: Loaded custom dataset
    """
    print("Loading custom dataset...")
    custom_data = DatasetDict()

    # Load the training data
    train_audio_files = [os.path.join("./custom_data/train", f) for f in os.listdir("./custom_data/train")]
    train_transcriptions = [os.path.join("./custom_data/transcriptions", f.replace(".wav", ".txt")) for f in os.listdir("./custom_data/train")]

    train_data = {
        "audio": train_audio_files,
        "sentence": [open(txt_file, "r").read().strip() for txt_file in train_transcriptions],
    }

    custom_data["train"] = Dataset.from_dict(train_data)

    # Load the test data (similar to train data structure)
    test_audio_files = [os.path.join("./custom_data/test", f) for f in os.listdir("./custom_data/test")]
    test_transcriptions = [os.path.join("./custom_data/transcriptions", f.replace(".wav", ".txt")) for f in os.listdir("./custom_data/test")]

    test_data = {
        "audio": test_audio_files,
        "sentence": [open(txt_file, "r").read().strip() for txt_file in test_transcriptions],
    }

    custom_data["test"] = Dataset.from_dict(test_data)

    # Cast the audio column to Audio format with 16kHz sampling rate
    custom_data = custom_data.cast_column("audio", Audio(sampling_rate=16000))

    print("Custom dataset loaded.")
    return custom_data

# Load dataset function
def load_dataset_function():
    """
    Loads the dataset and prepares it for preprocessing.
    :return: Loaded dataset
    """
    print("Loading dataset...")
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "mr", split="train+validation")
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "mr", split="test")
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    # Cast the audio column to Audio format with 16kHz sampling rate
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    print("Dataset loaded.")
    return common_voice

# Preprocess dataset function
def preprocess_data(common_voice, feature_extractor, tokenizer):
    """
    Pre processes the dataset by converting audio and text into input features.
    :param common_voice: The loaded dataset
    :param feature_extractor: WhisperFeatureExtractor to process audio
    :param tokenizer: WhisperTokenizer to process text
    :return: Preprocessed dataset
    """
    print("Preprocessing data...")
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = common_voice.map(lambda batch: prepare_dataset(batch), remove_columns=common_voice.column_names["train"], num_proc=4)
    print("Data preprocessed.")
    return common_voice

# Create the Whisper model
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.language = "marathi"
    model.generation_config.task = "transcribe"
    model.config.gradient_checkpointing = True
    model.generation_config.forced_decoder_ids = None
    return model

# Data Collator for speech-to-text with padding
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

# Compute metrics for evaluation (Word Error Rate)
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = pred.tokenizer.pad_token_id
    pred_str = pred.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = pred.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * evaluate.load("wer").compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Set up the trainer with all necessary arguments and configurations
def setup_trainer(common_voice, model, processor):
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
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    return trainer

# Main function to train and save the model
def train_and_save_model():
    # Load data
    common_voice = load_dataset_function()

    # Feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")

    # Preprocess data
    common_voice = preprocess_data(common_voice, feature_extractor, tokenizer)

    # Load model
    model = load_model()

    # Set up trainer
    trainer = setup_trainer(common_voice, model, processor)

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
    train_and_save_model()
    # Optionally, upload the model to Hugging Face
    # upload_model_to_huggingface(model)
