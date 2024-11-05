from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Define model ID
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Set up 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True  # Optional; set to True if you want to load in 8-bit as well
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Example usage
prompt = "hello how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU if available
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
