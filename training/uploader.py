from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from huggingface_hub import HfApi, upload_folder

checkpoint_dir = "./whisper-small-mr/checkpoint-100"
output_dir = "./models/whisper-small-mr-en-translation_v2"
repo_id = "pankaj-ag/whisper-medium-mr-en-translation-v2"

# Initialize Hugging Face API
# api = HfApi()

# api.create_repo(repo_id=repo_id, repo_type="model")

# # Load model, tokenizer, and feature extractor
# model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# # Save model, tokenizer, and feature extractor to the same directory
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# feature_extractor.save_pretrained(output_dir)

# Upload the complete model directory with all necessary files
upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned Whisper model with tokenizer and feature extractor"
)

print("Model uploaded successfully.")