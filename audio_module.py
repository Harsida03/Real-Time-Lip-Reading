"""
Audio capture module for real-time microphone input.
Requires: sounddevice, speech_recognition, numpy, portaudio (system library)
WARNING: This module will fail in environments without microphone/audio support.
"""

import speech_recognition as sr
import sounddevice as sd
import numpy as np
import queue
from typing import Optional
from fusion import AudioVisualFusion


class AudioCapture:
    """
    Captures audio from microphone and performs speech recognition.
    Requires microphone hardware and portaudio library.
    """
    
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        
    def stop_recording(self) -> Optional[str]:
        self.is_recording = False
        
        if not self.audio_data:
            return None
            
        audio_array = np.concatenate(self.audio_data, axis=0)
        
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        
        audio_instance = sr.AudioData(
            audio_bytes,
            sample_rate=self.sample_rate,
            sample_width=2
        )
        
        try:
            text = self.recognizer.recognize_google(audio_instance, show_all=False)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[AUDIO ERROR] Could not request results: {e}")
            return None
        except Exception as e:
            print(f"[AUDIO ERROR] Unexpected error: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO WARNING] {status}")
        
        if self.is_recording:
            self.audio_data.append(indata.copy())
    
    def get_audio_stream(self):
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32
        )


# Export AudioVisualFusion from fusion module
__all__ = ['AudioCapture', 'AudioVisualFusion']
