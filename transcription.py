import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import whisper
#hello

whisper_model = whisper.load_model("small")

# Load the Whisper model and processor from Hugging Face
def load_asr_model(modelName):
    processor = WhisperProcessor.from_pretrained(modelName)
    model = WhisperForConditionalGeneration.from_pretrained(modelName)
    return processor, model


def transcribe_audio(file_path, model, processor, language):
    # Load the audio file using librosa
    audio_array, sampling_rate = librosa.load(file_path, sr=16000)

    # Preprocess audio with WhisperProcessor
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features

    # Generate transcription using the fine-tuned model
    with torch.no_grad():
        generated_tokens = model.generate(input_features)
        transcription = processor.decode(generated_tokens[0], skip_special_tokens=True)

    return transcription

def translate_audio(file_path, model, processor, language):
    # Load the audio file using librosa
    audio_array, sampling_rate = librosa.load(file_path, sr=16000)

    # Preprocess audio with WhisperProcessor
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features

    # Generate transcription using the fine-tuned model
    with torch.no_grad():
        generated_tokens = model.generate(input_features, forced_decoder_ids=processor.get_decoder_prompt_ids(language="mr", task="translate"))
        transcription = processor.decode(generated_tokens[0], skip_special_tokens=True)

    return transcription

def translate_with_whisper(file_path, model, processor, language):
    # Transcribe using Whisper
    result = whisper_model.transcribe(file_path, task="translate", language = language)
    transcription = result['text']
    print(f"Transcription: {transcription}")
    return transcription

def transcribe_with_whisper(file_path, model, processor, language):
    # Transcribe using Whisper
    result = whisper_model.transcribe(file_path, task="transcribe", language = language)
    transcription = result['text']
    print(f"Transcription: {transcription}")
    return transcription
