from flask import Flask, jsonify, render_template, request, url_for
# import sounddevice as sd
# import scipy.io.wavfile
# import whisper
from gtts import gTTS
import os
from datetime import datetime

import whisper
from dotenv import load_dotenv # type: ignore
import os
from huggingface_hub import login
from mlx_lm import load, generate # type: ignore

# Load the environment variables from the .env file
load_dotenv()

# login to huggingface
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

# Initialize Flask app
app = Flask(__name__)

# # Load Whisper model
whisper_model = whisper.load_model("medium")

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

selected_language = 'hi'

language_configs = {
    "en": {
        "chatbot_instruction": "Please answer the following question in English:\n",
    },
    "hi": {
        "chatbot_instruction": "कृपया निम्नलिखित प्रश्न का उत्तर हिंदी में दें और हाइलाइट्स के लिए विशेष कीवर्ड जैसे * बोल्ड आदि से बचें:\n",
    },
    "mr": {
        "chatbot_instruction": "कृपया पुढील प्रश्नाचे उत्तर मराठीत द्या:\n",
    },
    "bn": {
        "chatbot_instruction": "দয়া করে নিচের প্রশ্নের উত্তর দিন মারাঠিতে:\n",
    },
}

OUTPUT_SAVE_PATH = "static/outputs"


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

    RECORDING_SAVE_PATH = "static/recordings"
    os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)

    audio_filename = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)

    file_name = f"{RECORDING_SAVE_PATH}/recording.wav"

    return file_name

def get_chatbot_response(input_text, language):
    instruction = language_configs[language]['chatbot_instruction']
    prompt = instruction + input_text

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            # {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them take best decisions for their crops and growth. Keep your responses brief and short until asked for details."},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=True)

    print("response generated", response)
    return {"content": response}

def transcribe_audio(file_path, language):
    result = whisper_model.transcribe(file_path, task="translate", language=language)
    return result.get("text", "")

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

    transcription = transcribe_audio(file_name, selected_language)
    timestamped_print("Audio transcribed", transcription)


    response = get_chatbot_response(transcription, selected_language)
    response_text = response['content']
    timestamped_print("Answer generated", response_text)


    audio_file_path = convert_to_audio(response_text, selected_language)
    timestamped_print("converted in audio", audio_file_path)


    # Print current time
    print(f"Query end time: {get_current_time()}")

    return jsonify({
        "user_input": transcription,
        "response_text": response_text,
        "audio_file_path": audio_file_path,
    })

if __name__ == "__main__":
    app.run(debug=True)
