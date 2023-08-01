import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class LLM:
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


class Generative(LLM):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

    @torch.inference_mode()
    async def generate_text(self, prompt: list):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        ret = await self.model.generate(input_ids, max_length=10000)
        return ret


class Embeddings(LLM):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        if 'instructor' in model_name:
            try:
                from InstructorEmbedding import INSTRUCTOR
            except ImportError as e:
                raise ImportError("Dependencies for InstructorEmbedding not found.")
            self.model = INSTRUCTOR(model_name)
            self.model = self.model.to(self.device)
        else:
            try:
                import sentence_transformers
            except ImportError as e:
                raise ImportError("Dependencies for sentence_transformers not found.")
            self.model = sentence_transformers.SentenceTransformer(model_name)
            self.model = self.model.to(self.device)

    @torch.inference_mode()
    async def encode(self, prompt: list) -> list:
        return self.model.encode(prompt).tolist()


