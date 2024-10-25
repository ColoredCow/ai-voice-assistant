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

# Route to render the HTML page with the recording UI
@app.route('/')
def index():
    return render_template('record.html')

# Flask route to record and transcribe audio
@app.route('/process-audio', methods=['POST'])
def record_audio_endpoint():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_data = request.files['audio_data']
    audio_bytes = audio_data.read()

    RECORDING_SAVE_PATH = "static/recordings"
    os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)

    audio_filename = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)

    file_name = f"{RECORDING_SAVE_PATH}/recording.wav"

    # Transcribe using Whisper
    result = whisper_model.transcribe(file_name, task="translate")
    transcription = result['text']
    print(f"Transcription: {transcription}")

    user_input = transcription
    response = get_chatbot_response(user_input)
    response_text = response['content']
    print(f"Response: {response_text}")

    OUTPUT_SAVE_PATH = "static/outputs"
    os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
    tts = gTTS(text=response_text, lang='mr', tld='co.in')
    tts.save(f"{OUTPUT_SAVE_PATH}/final-output.mp3")

    audio_file_path = url_for('static', filename='outputs/final-output.mp3')

    return jsonify({
        "user_input": user_input,
        "response_text": response_text,
        "audio_file_path": audio_file_path,
    })

if __name__ == "__main__":
    app.run(debug=True)
