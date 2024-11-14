import torch
from transformers import pipeline
import ollama

from ollama_utils import setup_model

model_id = "coloredcow/paani-1b-instruct-marathi"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

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

model_name = "paani"

setup_model(model_name)


def get_chatbot_response_old(input_text, language):
    instruction = language_configs[language]['chatbot_instruction']
    prompt = instruction + input_text

    if model_id == 'meta-llama/Llama-3.2-1B-Instruct' or model_id == 'coloredcow/paani-1b-instruct-marathi' or model_id == 'pankaj-ag/fine_tuned_model':
        print('inside model id check')
        messages = [
            # {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them take best decisions for their crops and growth. Keep your responses brief and short until asked for details."},
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=256,
            do_sample=False,
        )
        response = outputs[0]["generated_text"][-1]
        print("response from model......", response)
        return response['content']
    
    return None

def get_chatbot_response(input_text, language):
    instruction = language_configs[language]['chatbot_instruction']
    prompt = instruction + input_text

    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False  # Get full response, not in streaming mode
    )

    print(response['message'])

    # The response will be a list, and we need to extract the content
    if response:
        return response['message']['content']
    
    return "Sorry, I couldn't understand that."