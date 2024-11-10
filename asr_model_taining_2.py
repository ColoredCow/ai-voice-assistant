import os
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
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



