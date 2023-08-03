import torch
from threading import Thread
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from models.base import GenerativeLLM


class Vicuna(GenerativeLLM):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).half()
        self.model = self.model.to(self.device)
    
    def get_prompt(self, prompts: list) -> str:
        prompt = ''
        if self.tokenizer.eos_token == None:
            sep_token = '\n\n'
        else:
            sep_token = self.tokenizer.eos_token
        for message in prompts:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f'{content}{sep_token}'
            else: 
                prompt += f'{role}: {content}{sep_token}'
        prompt += "assistant:"
        return prompt

    @torch.inference_mode()
    async def generate_text(self, prompts: list, n: int, max_length: int) -> dict:
        results = {}
        compl_list = []
        prompt_tokens = 0
        completion_tokens = 0
        prompt = self.get_prompt(prompts)
        prompt_tokens += self.get_token_count(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        for _ in range(n):
            output_ids = self.model.generate(**input_ids, 
                                             max_length=max_length).tolist()
            gen_text = self.tokenizer.decode(output_ids[0][prompt_tokens :],
                                             skip_special_tokens=True,
                                             spaces_between_special_tokens=False)
            completion_tokens += self.get_token_count(gen_text)
            compl_list.append(gen_text)
        results['text'] = compl_list
        results['prompt_tokens'] = prompt_tokens
        results['completion_tokens'] = completion_tokens
        return results

    @torch.inference_mode()
    def generate_stream(self, prompts: list, max_length: int):
        prompt_tokens = 0
        prompt = self.get_prompt(prompts)
        prompt_tokens += self.get_token_count(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True})
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        generation_kwargs = dict(input_ids, streamer=streamer,
                                 max_new_tokens=max_length)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for i, token in enumerate(streamer):
            yield {
                "text": token,
                "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": i,
                },
            }