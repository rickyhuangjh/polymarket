import io
import os
import pyaudio
from google.cloud import speech
from google.cloud.speech import types
from google.cloud.speech import enums

# Set up the audio recording
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

def get_audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    return stream

def transcribe_stream():
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    stream = get_audio_stream()

    with stream:
        audio_generator = (
            types.StreamingRecognizeRequest(audio_content=content)
            for content in iter(lambda: stream.read(CHUNK), b"")
        )

        requests = iter(audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            for result in response.results:
                print("Transcript: {}".format(result.alternatives[0].transcript))

if __name__ == "__main__":
    transcribe_stream()


