import io
import os
import speech_recognition as sr

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import sounddevice 

from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel

class FasterWhisper():
    def __init__(self):
        self.device = "cuda"
        self.sample_rate = 16000
        self.sample_width = 2
        self.temp_file = NamedTemporaryFile().name 
        self.load_model()

    def load_model(self, model = "medium.en", compute_type = "float16"):
        self.model = WhisperModel(model, device = self.device, compute_type = compute_type)
        
    def transcribe(self, data):
        audio = sr.AudioData(data, sample_rate = self.sample_rate, sample_width = self.sample_width)
        wav = io.BytesIO(audio.get_wav_data())

        with open(self.temp_file, 'w+b') as f:
            f.write(wav.read())

        text = ""
                
        segments, info = self.model.transcribe(self.temp_file, condition_on_previous_text = False)
        for segment in segments:
            text += segment.text

        return text