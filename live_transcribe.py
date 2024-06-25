import whisper
import torch
import pyaudio
import numpy as np

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = whisper.load_model("medium", device=device)


# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
BUFFER_SIZE = 20  # Number of chunks to collect before transcribing

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording... Press Ctrl+C to stop.")

audio_buffer = []

try:
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(CHUNK)
        audio_buffer.append(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0)

        if len(audio_buffer) >= BUFFER_SIZE:
            # Concatenate buffer into a single audio array
            audio_input = np.concatenate(audio_buffer)
            audio_buffer = []

            # Transcribe audio data using Whisper
            result = model.transcribe(audio_input)
            print(result["text"])

except KeyboardInterrupt:
    print("Recording stopped.")

finally:
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
