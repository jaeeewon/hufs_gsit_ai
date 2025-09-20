import requests
import json

from .abs_class.llm import AbsCallLLM
from .config import QWEN_URL


class LLMCall(AbsCallLLM):

    def __init__(self):
        pass

    def call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {"text": prompt}

        try:
            response = requests.post(QWEN_URL, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                return response.json()["generated_text"]
        except requests.exceptions.RequestException as e:
            return {"error": response.text if response else str(e)}
