from flask import Flask, jsonify, render_template, request, url_for
# import sounddevice as sd
# import scipy.io.wavfile
# import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import librosa
from gtts import gTTS
import os
from datetime import datetime
from dotenv import load_dotenv
import os
from huggingface_hub import login
# from mlx_lm import load, generate

# Load the environment variables from the .env file
load_dotenv()

# login to huggingface
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

def get_current_time():
    """Get the current time"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    return formatted_time

# Initialize Flask app
app = Flask(__name__)

# # Load Whisper model
# whisper_model = whisper.load_model("base")

# # meta-llama/Llama-3.2-1B-Instruct
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Load the model with 4-bit quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     torch_dtype=torch.float16,  # Setting this helps with speed and memory
# )

# model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# pipe = pipeline(model='sarvamai/shuka_v1', trust_remote_code=True, device=0, torch_dtype='bfloat16')

selected_language = 'hi'
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

# # Record audio function
# def record_audio(duration=5, sample_rate=44100):
#     """Record audio for the specified duration."""
#     print(f"Recording for {duration} seconds...")
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
#     sd.wait()  # Wait until the recording is finished
#     return audio_data, sample_rate

# # Save audio file
# def save_audio(file_name, audio_data, sample_rate):
#     """Save recorded audio to a file."""
#     scipy.io.wavfile.write(file_name, sample_rate, audio_data)

# def get_chatbot_response(input_text, language):
#     instruction = language_configs[language]['chatbot_instruction']
#     prompt = instruction + input_text

#     if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
#         messages = [
#             # {"role": "system", "content": "You are a chatbot designed to help Indian farmers on any agriculture related questions they have. Be a helpful guide and friend to empower them take best decisions for their crops and growth. Keep your responses brief and short until asked for details."},
#             {"role": "user", "content": prompt},
#         ]
#         prompt = tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#     response = generate(model, tokenizer, prompt=prompt, verbose=True)
#     return response


# Route to render the HTML page with the recording UI
@app.route('/')
def index():
    return render_template('record.html')

# # Flask route to record and transcribe audio
# @app.route('/process-audio', methods=['POST'])
# def record_audio_endpoint():
    # Print current time
    print(f"Query start time: {get_current_time()}")

    # if 'audio_data' not in request.files:
    #     return jsonify({"error": "No audio file uploaded"}), 400

    # audio_data = request.files['audio_data']
    # audio_bytes = audio_data.read()

    # RECORDING_SAVE_PATH = "static/recordings"
    # os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)

    # audio_filename = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
    # with open(audio_filename, "wb") as f:
    #     f.write(audio_bytes)

    # file_name = f"{RECORDING_SAVE_PATH}/recording.wav"

    # # Transcribe using Whisper
    # result = whisper_model.transcribe(file_name, task="translate")
    # transcription = result['text']
    # print(f"Transcription: {transcription}")

    # user_input = transcription
    user_input = "tell me about crop seasons in India"
    # user_input = "hello how are you?"
    # response = get_chatbot_response(user_input, selected_language)
    # response_text = response['content']
    response_text = get_chatbot_response(user_input, selected_language)

    OUTPUT_SAVE_PATH = "static/outputs"
    os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
    tts = gTTS(text=response_text, lang=selected_language, tld='co.in')
    tts.save(f"{OUTPUT_SAVE_PATH}/final-output.mp3")

    audio_file_path = url_for('static', filename='outputs/final-output.mp3')

    # Print current time
    print(f"Query end time: {get_current_time()}")

    return jsonify({
        "user_input": user_input,
        "response_text": response_text,
        "audio_file_path": audio_file_path,
    })

# @app.route('/process-audio', methods=['POST'])
# def process_audio():
#     print(f"Query start time: {get_current_time()}")

#     if 'audio_data' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     audio_data = request.files['audio_data']
#     audio_bytes = audio_data.read()

#     RECORDING_SAVE_PATH = "static/recordings"
#     os.makedirs(RECORDING_SAVE_PATH, exist_ok=True)

#     audio_filename = os.path.join(RECORDING_SAVE_PATH, "recording.wav")
#     with open(audio_filename, "wb") as f:
#         f.write(audio_bytes)

#     file_name = f"{RECORDING_SAVE_PATH}/recording.wav"

#     audio, sr = librosa.load(file_name, sr=16000)
#     turns = [
#         {'role': 'system', 'content': 'Respond naturally and informatively.'},
#         {'role': 'user', 'content': audio}
#     ]
#     outputs = pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=512)
#     print('outputs..................')
#     print(outputs)
#     response = outputs[0]["generated_text"][-1]
#     response_text = response['content']
#     print(f"Query end time: {get_current_time()}")
#     return jsonify({
#         "user_input": user_input,
#         "response_text": response_text,
#         # "audio_file_path": audio_file_path,
#     })

if __name__ == "__main__":
    app.run(debug=True)
