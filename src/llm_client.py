import os
from openai import OpenAI
from src import config


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
        )
        self.model = config.LLM_MODEL

    def chat(self, messages: list[dict], temperature: float = 0.7, stream: bool = False):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )

    def chat_stream(self, messages: list[dict], temperature: float = 0.7):
        response = self.chat(messages, temperature=temperature, stream=True)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
