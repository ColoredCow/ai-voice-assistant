from flask import Flask, render_template
import sounddevice as sd
import scipy.io.wavfile
import whisper
import torch
from transformers import pipeline
from gtts import gTTS
import os

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("medium")

# meta-llama/Llama-3.2-1B-Instruct
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

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

def get_chatbot_response(input_text):
    instruction = "कृपया पुढील प्रश्नाचे उत्तर मराठीत द्या:\n" # marathi
    # instruction = "দয়া করে নিচের প্রশ্নের উত্তর দিন মারাঠিতে:\n" # bengali
    prompt = instruction + input_text
    messages = [
        {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them take best decisions for their crops and growth. Keep your responses brief and short until asked for details. Translate all your response to Marathi."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

# Flask route to record and transcribe audio
@app.route('/record', methods=['GET'])
def record_audio_endpoint():
    duration = 5
    file_name = "static/audio-input.wav"

    # Record and save the audio
    audio_data, sample_rate = record_audio(duration=duration)
    save_audio(file_name, audio_data, sample_rate)

    # Transcribe using Whisper
    result = whisper_model.transcribe(file_name, task="translate")
    transcription = result['text']
    print(f"Transcription: {transcription}")

    user_input = transcription
    response = get_chatbot_response(user_input)

    response_text = response['content']

    print(f"Response: {response_text}")

    tts = gTTS(text=response_text, lang='mr', tld='co.in')
    tts.save("static/final-output.mp3")

    return render_template('record.html', user_input=user_input, response_text=response_text)

if __name__ == "__main__":
    app.run(debug=True)
