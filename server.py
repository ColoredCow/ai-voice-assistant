from datetime import datetime
from flask import Flask, jsonify, render_template, request, url_for
import sounddevice as sd
import scipy.io.wavfile
import whisper
import torch
from transformers import pipeline
from gtts import gTTS
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
whisper_model = whisper.load_model("medium")
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Constants
RECORDING_SAVE_PATH = "static/recordings"
OUTPUT_SAVE_PATH = "static/outputs"

# Ensure directories exist
os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)

def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n\n")
    print( f"[{timestamp}]", *args, **kwargs)
    print("\n\n")


def record_audio(duration=5, sample_rate=44100):
    """Record audio for the specified duration."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    return audio_data, sample_rate

def save_audio(file_name, audio_data, sample_rate):
    """Save recorded audio to a file."""
    scipy.io.wavfile.write(file_name, sample_rate, audio_data)

def transcribe_audio(file_path):
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(file_path, task="translate", language="mr")
    return result.get("text", "")

def get_chatbot_response(input_text):
    """Generate response using language model."""
    prompt = f"कृपया पुढील प्रश्नाचे उत्तर मराठीत द्या:\n{input_text}"
    messages = [
        {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them to make the best decisions for their crops and growth. Keep your responses brief and short until asked for details. Translate all your response to Marathi."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages, max_new_tokens=256)
    print("outputs", outputs)
    return outputs[0]["generated_text"]

def generate_audio_response(text, file_path):
    """Convert text response to audio."""
    tts = gTTS(text=text, lang='mr', tld='co.in')
    tts.save(file_path)

@app.route('/')
def index():
    return render_template('record.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save the uploaded audio
    audio_data = request.files['audio_data']
    audio_file_path = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
    audio_data.save(audio_file_path)

    timestamped_print("audio saved")
    # Transcribe audio to text
    transcription = transcribe_audio(audio_file_path)
    if not transcription:
        return jsonify({"error": "Transcription failed"}), 500

    timestamped_print("audio transcribed", transcription)
    # Get chatbot response based on transcription
    chatbot_response = get_chatbot_response(transcription)
    timestamped_print("chatbot_response", chatbot_response)
    response_text = chatbot_response[2]['content']
    timestamped_print("got response from chatbot", response_text)
    # Convert response text to audio
    audio_response_path = os.path.join(OUTPUT_SAVE_PATH, "final-output.mp3")
    generate_audio_response(response_text, audio_response_path)
    timestamped_print("generated audio responses")

    return jsonify({
        "transcription": transcription,
        "response_text": response_text,
        "audio_file_path": url_for('static', filename='outputs/final-output.mp3')
    })

if __name__ == "__main__":
    app.run(debug=True)
