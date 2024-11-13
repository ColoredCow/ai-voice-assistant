import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Local path to your saved model
model_path = "./training/models/fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# model_id = "coloredcow/paani-1b-instruct-marathi"

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

selected_language = 'en'
language_configs = {
    "en": {
        "chatbot_instruction": "Please answer the following question in English:\n",
    },
    "hi": {
        "chatbot_instruction": "कृपया निम्नलिखित प्रश्न का उत्तर हिंदी में दें:\n",
    },
    "mr": {
        "chatbot_instruction": "कृपया पुढील प्रश्नाचे उत्तर मराठीत द्या:\n",
    },
    "bn": {
        "chatbot_instruction": "দয়া করে নিচের প্রশ্নের উত্তর দিন মারাঠিতে:\n",
    },
}

def get_chatbot_response(input_text, language):
    instruction = language_configs[language]['chatbot_instruction']
    prompt = instruction + input_text

    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=False,
    )
    response = outputs[0]["generated_text"]
    return response

# Interactive loop for user input
print("Chatbot is ready! Type 'exit' to quit.\n")

while True:
    # Prompt the user for input
    user_input = input("You: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break

    # Get and print the chatbot response
    response_text = get_chatbot_response(user_input, selected_language)
    print("Chatbot:", response_text)
