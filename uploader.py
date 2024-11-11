from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from huggingface_hub import upload_folder

checkpoint_dir = "./whisper-small-mr/checkpoint-100"
output_dir = "./whisper-small-mr-finetuned-marathi"
repo_id = "pankaj-ag/whisper-small-mr-finetuned-marathi"

# Load model, tokenizer, and feature extractor
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# Save model, tokenizer, and feature extractor to the same directory
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)

# Upload the complete model directory with all necessary files
upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned Whisper model with tokenizer and feature extractor"
)
