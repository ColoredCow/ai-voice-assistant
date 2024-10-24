from flask import Flask, jsonify
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import whisper
import torch
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("base")

# meta-llama/Llama-3.2-1B-Instruct
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load Hugging Face conversational model
# model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # Set pad_token_id to the EOS token for padding
# chatbot_model.config.pad_token_id = tokenizer.eos_token_id

# Record audio function
def record_audio(duration=5, sample_rate=44100):
    """Record audio for the specified duration."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    return audio_data, sample_rate

# Save audio file
def save_audio(file_name, audio_data, sample_rate):
    """Save recorded audio to a file."""
    scipy.io.wavfile.write(file_name, sample_rate, audio_data)

# # Function to get chatbot response
# def get_chatbot_response(input_text, chat_history_ids=None):
#     # Tokenize the new user input and add the end-of-sequence token
#     new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

#     # Append the new user input to the chat history (if any)
#     if chat_history_ids is not None:
#         input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
#     else:
#         input_ids = new_input_ids

#     # Generate a response, with a max length of 1000 tokens
#     chat_history_ids = chatbot_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # Decode the last generated response (excluding previous chat history)
#     response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

#     return response, chat_history_ids

def get_chatbot_response(input_text):
    messages = [
        {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them take best decisions for their crops and growth. Keep your responses brief and short until asked for details."},
        {"role": "user", "content": input_text},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

# Flask route to record and transcribe audio
@app.route('/record', methods=['GET'])
def record_audio_endpoint():
    duration = 3  # Record for 3 seconds
    file_name = "output.wav"

    # Record and save the audio
    audio_data, sample_rate = record_audio(duration=duration)
    save_audio(file_name, audio_data, sample_rate)

    # Transcribe using Whisper
    result = whisper_model.transcribe(file_name)
    transcription = result['text']
    print(f"Transcription: {transcription}")

    user_input = transcription
    # user_input = 'Hi there, how are you?'
    response = get_chatbot_response(user_input)

    return jsonify({
        # "message": f"Audio recorded and saved to {file_name}!",
        "1. user input": user_input,
        "2. model response": response,
    })

if __name__ == "__main__":
    app.run(debug=True)
