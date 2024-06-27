from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def ping():
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Who won the world series in 2020?"}],
            max_tokens=5
        )

        end_time = time.time()
        print(f'Ping: {end_time-start_time:.2f} seconds')

    except Exception as e:
        print(f"Failed to reach the API: {e}")
