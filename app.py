from flask import Flask, jsonify, render_template, request, url_for
from gtts import gTTS
import os
from datetime import datetime
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transcription import load_asr_model, translate_audio, translate_with_base_whisper
from chatbot import get_chatbot_response

# Load the environment variables from the .env file
load_dotenv()

# login to huggingface
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

RECORDING_SAVE_PATH = "static/recordings"
OUTPUT_SAVE_PATH = "static/outputs"
ASR_MODEL_NAME = 'pankaj-ag/whisper-small-mr-en-translation'

processor, asr_model = load_asr_model(ASR_MODEL_NAME)

selected_language = 'en'

# Initialize Flask app
app = Flask(__name__)

def get_current_time():
    """Get the current time"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    return formatted_time

def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n\n")
    print( f"[{timestamp}]", *args, **kwargs)
    print("\n\n")

def save_audio(files):
    if 'audio_data' not in files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_data = request.files['audio_data']
    audio_bytes = audio_data.read()
    os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)

    audio_filename = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)

    file_name = f"{RECORDING_SAVE_PATH}/recording.wav"

    return file_name

def convert_to_audio(text, language):
    os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
    tts = gTTS(text=text, lang=language, tld='co.in')
    tts.save(f"{OUTPUT_SAVE_PATH}/final-output.mp3")
    audio_file_path = url_for('static', filename='outputs/final-output.mp3')
    return audio_file_path

# Route to render the HTML page with the recording UI
@app.route('/')
def index():
    return render_template('record.html')

# Flask route to record and transcribe audio
@app.route('/process-audio', methods=['POST'])
def record_audio_endpoint():
    # Print current time
    print(f"Query start time: {get_current_time()}")

    file_name = save_audio(request.files)
    timestamped_print("Audio file saved")

    transcription = translate_audio(file_name, asr_model, processor, selected_language)
    timestamped_print("Audio translate_audio", transcription)

    user_input = transcription
    response_text = get_chatbot_response(user_input, selected_language)

    audio_file_path = convert_to_audio(response_text, selected_language)
    timestamped_print("converted in audio", audio_file_path)

    # Print current time
    print(f"Query end time: {get_current_time()}")

    return jsonify({
        "user_input": user_input,
        "recorded_audio_path": file_name,
        "response_text": response_text,
        "model_id": "",
        "audio_file_path": audio_file_path,
    })

if __name__ == "__main__":
    app.run(debug=True)