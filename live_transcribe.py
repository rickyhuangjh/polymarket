import sounddevice as sd
import queue
import sys
from google.cloud import speech

# Audio recording parameters
RATE = 16000
CHANNELS = 1

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def transcribe_stream():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
        audio_generator = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in iter(q.get, None)
        )

        requests = iter(audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            for result in response.results:
                print("Transcript: {}".format(result.alternatives[0].transcript))

if __name__ == "__main__":
    transcribe_stream()
