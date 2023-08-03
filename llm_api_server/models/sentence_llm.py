import torch
try:
    import sentence_transformers
except ImportError as e:
    raise ImportError("Dependencies for sentence_transformers not found.")

from models.base import EmbeddingsLLM


class SentenceLLM(EmbeddingsLLM):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.model = self.model.to(self.device)

    @torch.inference_mode()
    async def encode(self, prompt: list) -> list:
        return self.model.encode(prompt).tolist()

