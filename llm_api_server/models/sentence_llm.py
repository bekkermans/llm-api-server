import torch
import sentence_transformers
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer
from models.base import EmbeddingsLLM


class SentenceLLM(EmbeddingsLLM):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = kwargs.get('device', None)
        if 'instructor' in self.model_name:
            self.model = INSTRUCTOR(self.model_name, device=device)
        else:
            self.model = sentence_transformers.SentenceTransformer(self.model_name,
                                                                    device=device)
        self.device = self.model.device

    def get_token_count(self, prompt: str) -> int:
        count = self.tokenizer(prompt, return_length=True, return_tensors='np')
        return int(count['length'][0])

    @torch.inference_mode()
    async def encode(self, prompt: list) -> list:
        return self.model.encode(prompt).tolist()