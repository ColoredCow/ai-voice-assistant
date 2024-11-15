from flask import Flask, jsonify, render_template, request, Response, stream_with_context, url_for
from gtts import gTTS
import os
from datetime import datetime
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transcription import load_asr_model, translate_audio, translate_with_base_whisper
from chatbot import get_chatbot_response, get_chatbot_response_stream

# Load the environment variables from the .env file
load_dotenv()

# login to huggingface
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

RECORDING_SAVE_PATH = "static/recordings"
OUTPUT_SAVE_PATH = "static/outputs"
ASR_MODEL_NAME = 'pankaj-ag/whisper-medium-mr-en-translation-v2'

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

@app.route('/process-audio', methods=['POST'])
def process_audio():
    print(f"Query start time: {get_current_time()}")

    files = request.files
    stream = request.form.get('stream', 'true')
    stream = request.args.get('stream', 'false').lower()
    if stream == 'true':
        stream = True
    else:
        stream = False

    lang = request.form.get("lang", selected_language)
    

    file_name = save_audio(files)
    timestamped_print("Audio file saved")

    transcription = translate_audio(file_name, asr_model, processor, lang)
    timestamped_print("Audio translate_audio", transcription)

    user_input = transcription

    chat_bot_answer = ""

    if stream == False:
        chat_bot_answer = get_chatbot_response(user_input, lang)

    # Return metadata
    return jsonify({
        "user_input": user_input,
        "recorded_audio_path": file_name,
        "model_id": "Paani - llama3.2",
        "audio_file_path": '',
        "chat_bot_answer": chat_bot_answer
    })


# Route to stream chatbot response
@app.route('/stream-response', methods=['GET'])
def stream_response():
    user_input = request.args.get('user_input')
    lang = request.args.get('lang', selected_language)

    def generate_streamed_response():
        for chunk in get_chatbot_response_stream(user_input, lang):
            yield f"data:{chunk['message']['content']}\n\n"
        
        yield "data: [END OF RESPONSE]\n\n"

    return Response(stream_with_context(generate_streamed_response()), content_type='text/event-stream')

@app.route('/process-tts', methods=['GET'])
def get_audio_from_text():
    text = request.args.get('text') 
    audio_file_path = convert_to_audio(text, selected_language)

    return jsonify({
        "audio_file_path": audio_file_path,
    })

if __name__ == "__main__":
    app.run(debug=True)