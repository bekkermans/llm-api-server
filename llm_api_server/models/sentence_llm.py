import torch
import sentence_transformers
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer
from models.base import EmbeddingsLLM
from typing import List


class SentenceLLM(EmbeddingsLLM):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = kwargs.get('device_map', None)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if 'instructor' in self.model_name:
            self.model = INSTRUCTOR(self.model_name)
        else:
            self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.model = self.model.to(self.device)

    def get_token_count(self, prompt: str) -> int:
        count = self.tokenizer(prompt, return_length=True, return_tensors='np')
        return int(count['length'][0])

    @torch.inference_mode()
    async def encode(self, prompt: list) -> list:
        if all(isinstance(token, (list, int)) for token in prompt):
            # Convert input to string if a sentence is already tokenized.
            # This is a limitation of sentence_transformers. It allows only 
            # string input.
            prompt = self.tokenizer.batch_decode(prompt, skip_special_tokens=True)
        return self.model.encode(prompt).tolist()