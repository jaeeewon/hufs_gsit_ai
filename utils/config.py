from dotenv import load_dotenv
import os

load_dotenv("../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QWEN_URL = os.getenv("QWEN_URL", "http://127.0.0.1:8000/generate")
