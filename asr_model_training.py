import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from huggingface_hub import HfApi, login
import librosa
from transformers import WhisperTokenizer
from evaluate import load as load_metric

# Define the model name at the top-level
MODEL_NAME = "./whisper_finetuned_marathi_paani"

# Hugging Face login using your token (set this in the environment variable 'HUGGINGFACE_TOKEN')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

# Load and Preprocess Dataset (combine common_voice with your custom dataset)
def load_and_preprocess_data(custom_audio_files=None, custom_transcriptions=None):
    """
    Loads and preprocesses the data, optionally including client data if provided.

    :param custom_audio_files: List of paths to custom audio files
    :param custom_transcriptions: List of transcriptions corresponding to the custom audio files
    :return: Combined dataset of Common Voice and client data (if available)
    """
    # Load the Common Voice Marathi dataset
    print("start loading")
    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "mr")
    print("end loading")

    # Check if custom client data is provided; if not, set them to empty lists
    if custom_audio_files is None or custom_transcriptions is None or len(custom_audio_files) == 0 or len(custom_transcriptions) == 0:
        print("No custom client data provided. Proceeding with Common Voice dataset only.")
        custom_audio_data = Dataset.from_dict({
            'audio': [],
            'transcription': []
        })
    else:
        # Load your custom dataset if provided
        print(f"Custom client data provided: {len(custom_audio_files)} audio files.")
        custom_audio_data = Dataset.from_dict({
            'audio': custom_audio_files,  # Paths to custom audio files
            'transcription': custom_transcriptions  # Corresponding transcriptions
        })

    # Combine datasets (ensure the data is formatted properly)
    full_dataset = DatasetDict({
        'train': common_voice['train'].concatenate(custom_audio_data),
        'validation': common_voice['validation']
    })

    # Preprocess the audio files (make them consistent for Whisper)
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    def preprocess(batch):
        audio_array, sampling_rate = librosa.load(batch["audio"], sr=16000)
        batch["input_features"] = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features[0]
        return batch

    # Apply preprocessing
    full_dataset = full_dataset.map(preprocess, remove_columns=["audio"])

    return full_dataset

# Load the pretrained Whisper model
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    return model

# Training script
def train_model(custom_audio_files=None, custom_transcriptions=None):
    # Load data
    dataset = load_and_preprocess_data(custom_audio_files, custom_transcriptions)

    # Load the model
    model = load_model()

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_NAME,  # Use the top-level variable for model name
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        warmup_steps=500,
        save_steps=1000,
        logging_dir='./logs',
        num_train_epochs=3,
        fp16=True,  # Mixed precision training for speed
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

    # Metric for evaluation
    wer_metric = load_metric("wer")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")

    def compute_metrics(pred):
        preds = pred.predictions
        labels = pred.label_ids
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Trainer setup
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_NAME)

    return model

# Push model to Hugging Face Hub
def upload_model_to_huggingface(model, model_name):
    model.push_to_hub(model_name)
    print(f"Model uploaded to Hugging Face under the name: {model_name}")

# Check and update Hugging Face repository
def check_and_update_huggingface(model, model_name):
    api = HfApi()
    try:
        # Check if the model already exists on Hugging Face
        repo_info = api.repo_info(model_name)
        print(f"Model '{model_name}' already exists on Hugging Face. Updating...")
        model.push_to_hub(model_name, private=True)
        print(f"Model '{model_name}' updated.")
    except Exception as e:
        # If the model does not exist, create a new repo and upload it
        print(f"Model '{model_name}' does not exist. Creating new repo...")
        api.create_repo(model_name, private=True)
        model.push_to_hub(model_name, private=True)
        print(f"Model '{model_name}' created and uploaded.")

# Main entry point to train and upload
def start_training(custom_audio_files=None, custom_transcriptions=None):
    model = train_model(custom_audio_files, custom_transcriptions)
    check_and_update_huggingface(model, MODEL_NAME)

if __name__ == "__main__":
    custom_audio_files = None 
    custom_transcriptions = None
    # custom_audio_files = ["path/to/your/file1.wav", "path/to/your/file2.wav"]
    # custom_transcriptions = ["transcription 1", "transcription 2"]

    # Start training and uploading process
    start_training(custom_audio_files, custom_transcriptions)