from flask import Flask, jsonify
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import whisper
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load Hugging Face conversational model
model_name = "microsoft/DialoGPT-medium"
chatbot_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Set pad_token_id to the EOS token for padding
chatbot_model.config.pad_token_id = tokenizer.eos_token_id

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

# Function to get chatbot response
def get_chatbot_response(input_text, chat_history_ids=None):
    # Tokenize the new user input and add the end-of-sequence token
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history (if any)
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    # Generate a response, with a max length of 1000 tokens
    chat_history_ids = chatbot_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last generated response (excluding previous chat history)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

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

    # Initialize conversation
    chat_history_ids = None

    user_input = transcription
    response, chat_history_ids = get_chatbot_response(user_input, chat_history_ids)

    return jsonify({
        # "message": f"Audio recorded and saved to {file_name}!",
        "1. user input": user_input,
        "2. model response": response,
    })

if __name__ == "__main__":
    app.run(debug=True)
