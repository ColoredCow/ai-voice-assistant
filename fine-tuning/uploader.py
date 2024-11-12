from huggingface_hub import upload_folder

model_dir = "./fine_tuned_model"
repo_id = "coloredcow/paani-1b-instruct-marathi"

# Upload the complete model directory with all necessary files
upload_folder(
  folder_path=model_dir,
  repo_id=repo_id,
  repo_type="model",
  commit_message="init for model"
)

print("Model uploaded successfully.")
