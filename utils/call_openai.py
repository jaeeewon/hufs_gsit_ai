from .abs_class.llm import AbsCallLLM
from .config import OPENAI_API_KEY
from openai import OpenAI

supported_models = ["gpt-4o", "gpt-4.1-mini"]

client = OpenAI(api_key=OPENAI_API_KEY)


class LLMCall(AbsCallLLM):

    def __init__(self, model_name: str = supported_models[1]):
        if model_name not in supported_models:
            raise ValueError(f"model {model_name} not supported")
        self._model_name = model_name

    def call(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=0,
            max_tokens=1024,
            timeout=20,
        )

        return response.choices[0].message.content
