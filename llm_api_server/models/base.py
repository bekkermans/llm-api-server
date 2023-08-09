from abc import ABC, abstractmethod
from typing import Generator
from api_spec import ChatCompletionsRequest
import torch.cuda
from transformers import AutoTokenizer


class EmbeddingsLLM(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def get_token_count(self, prompt: str) -> int:
        count = self.tokenizer(prompt, return_length=True, return_tensors='np')
        return int(count['length'][0])

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