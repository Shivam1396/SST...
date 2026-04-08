import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME   = os.getenv("MODEL_NAME", "<your-model>")
HF_TOKEN     = os.getenv("HF_TOKEN")   # NO default for this one

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Stdout logs MUST follow this exact format:
print('[START] {"task": "easy"}')
print('[STEP]  {"action": ..., "reward": ..., "state": ...}')
print('[END]   {"score": 0.85}')