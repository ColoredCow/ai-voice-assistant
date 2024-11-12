from flask import Flask, jsonify, render_template, request, url_for
import torch
from transformers import pipeline
from gtts import gTTS
import os
from datetime import datetime
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transcription import load_asr_model, translate_with_base_whisper, translate_audio

# Load the environment variables from the .env file
load_dotenv()

# login to huggingface
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

RECORDING_SAVE_PATH = "static/recordings"
OUTPUT_SAVE_PATH = "static/outputs"
ASR_MODEL_NAME = 'pankaj-ag/whisper-small-mr-en-translation'

processor, asr_model = load_asr_model(ASR_MODEL_NAME)

def get_current_time():
    """Get the current time"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    return formatted_time

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model

model_id = "coloredcow/paani-1b-instruct-marathi"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


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
        "model_id": model_id,
        "audio_file_path": audio_file_path,
    })

if __name__ == "__main__":
    app.run(debug=True)