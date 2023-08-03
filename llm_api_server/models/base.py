from abc import abstractmethod
import torch.cuda
from transformers import AutoTokenizer


class EmbeddingsLLM:
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


class GenerativeLLM:
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
    def generate_text(self,):
        pass