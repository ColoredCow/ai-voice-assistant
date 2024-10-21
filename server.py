from flask import Flask, jsonify
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import whisper

# Initialize Flask app
app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("base")

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

    return jsonify({
        "message": f"Audio recorded and saved to {file_name}!",
        "transcription": transcription
    })

if __name__ == "__main__":
    app.run(debug=True)
