import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
import wave
import os
import time
from datetime import datetime
from ping import ping
import whisper
import asyncio
from collections import deque
import sounddevice as sd
import soundfile as sf
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

load_dotenv()

client = OpenAI()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
whisper_model = whisper.load_model('large')



BIDEN_PATH = 'biden.wav'
waveform, original_sample_rate = torchaudio.load(BIDEN_PATH)

new_sample_rate = 16000
transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
resampled_waveform = transform(waveform)

# Save the resampled audio (optional)
torchaudio.save("biden.wav", resampled_waveform, new_sample_rate)

USE_LOCAL_MODEL = True
    
# Constants
BUFFER_SECONDS = 4
APPEND_SECONDS = 1
API_SLEEP_SECONDS = 4
SAMPLE_RATE = 16000  # Adjust according to your needs

# Calculate the number of samples
buffer_size = BUFFER_SECONDS * SAMPLE_RATE
append_size = APPEND_SECONDS * SAMPLE_RATE

# Initialize the buffer
buffer = np.zeros(buffer_size, dtype=np.float32)

last_saved = 0

async def save_buffer_to_file():
    global buffer
    global last_saved
    prev_time = time.time()

    cur_filename = '0.wav'
    
    with sf.SoundFile(cur_filename, mode='w', samplerate=SAMPLE_RATE, channels=1, format='WAV') as file:
        while True:
            # Save the buffer to a file
            #cur_filename = f'{last_saved}.wav'
            file.write(buffer)
            last_saved = (last_saved + 1) % 8

            print(len(buffer))


            verification = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={'device': 'cuda'})

            score, prediction = verification.verify_files(cur_filename, BIDEN_PATH)

            print(score, 'BIDEN' if prediction == 1 else 'NOT BIDEN')
            
            
            
            start_time = time.time()
            transcription = transcribe_audio(buffer)
            # ping()
            end_time = time.time()
            cur_time = time.time()
            # print(f'time: {end_time-start_time:.2f}')
            print(f'{datetime.now().strftime('%M:%S')} {cur_time-prev_time:.2f}s', transcription)
            prev_time = cur_time

def audio_callback(indata, frames, timex, status):
    global buffer
    if status:
        print(status)
    
    # Append the new data
    buffer = np.append(buffer, indata[:, 0])
    
    # Pop the oldest data to maintain the buffer size
    if len(buffer) > buffer_size:
        buffer = buffer[-buffer_size:]

def transcribe_audio(filepath):
    if USE_LOCAL_MODEL:
        result = whisper_model.transcribe(buffer)
        return result['text']

    with open(filepath, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model='whisper-1')
        return response.text

async def double_transcribe():
    pass

async def handle_transcription():
    global last_saved
    prev_time = time.time()
    while True:
        filepath = f'{last_saved}.wav'
        transcription = transcribe_audio(filepath)
        cur_time = time.time()
        print(f'{datetime.now().strftime('%M:%S')} {cur_time-prev_time:.2f}s', transcription)
        prev_time = cur_time
        await asyncio.sleep(API_SLEEP_SECONDS)


async def main():
    save_task = asyncio.create_task(save_buffer_to_file())


    # Open a stream to record audio
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=append_size):
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(APPEND_SECONDS)
        except KeyboardInterrupt:
            print("Recording stopped.")
            save_task.cancel()
            await save_task  # Ensure the task is properly cancelled

if __name__ == "__main__":
    asyncio.run(main())



