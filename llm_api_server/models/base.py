from abc import ABC, abstractmethod
from typing import Generator
from api_spec import ChatCompletionsRequest


class EmbeddingsLLM(ABC):
    @abstractmethod
    def get_token_count(self, prompt: str) -> int:
        pass

    @abstractmethod
    def encode(self, prompt: list) -> list:
        pass


class GenerativeLLM(ABC):
    @abstractmethod
    def get_token_count(self, prompt: str) -> int:
        pass

    @abstractmethod
    def get_prompt(self, prompts: list) -> str:
        pass

    @abstractmethod
    async def generate_text(self, request: ChatCompletionsRequest) -> dict:
        pass

    @abstractmethod
    async def generate_stream(self, request: ChatCompletionsRequest) -> Generator:
        pass