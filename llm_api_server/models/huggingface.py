import torch
from abc import abstractmethod
from api_spec import ChatCompletionsRequest
from threading import Thread
from typing import AsyncGenerator
from transformers import (AutoModelForCausalLM, 
                          TextIteratorStreamer, 
                          AutoTokenizer, 
                          GenerationConfig)
from models.base import GenerativeLLM


class LLAMA2BASE(GenerativeLLM):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        params_override_dict = {
                "torch.bfloat16": torch.bfloat16,
                "torch.half": torch.half,
                "torch.float": torch.float,
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
            }

        for key, value in kwargs.items():
            if value in params_override_dict:
                kwargs[key] = params_override_dict[value]

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        if kwargs.get('device_map', None) is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
        else:
            self.device = kwargs.get('device_map')

    @abstractmethod
    def get_prompt(self, prompts: list) -> str:
        pass

    def get_token_count(self, prompt: str) -> int:
        count = self.tokenizer(prompt, return_length=True, return_tensors='np')
        return int(count['length'][0])

    @torch.inference_mode()
    async def generate_text(self, request: ChatCompletionsRequest) -> dict:
        results = {}
        compl_list = []
        prompt_tokens = 0
        completion_tokens = 0
        prompt = self.get_prompt(request.messages)
        prompt_tokens += self.get_token_count(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        generation_config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature >= 1e-3,
            temperature=request.temperature,
            top_p=request.top_p)

        for _ in range(request.n):
            output_ids = self.model.generate(**input_ids, 
                                generation_config=generation_config).tolist()
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
    async def generate_stream(self, request: ChatCompletionsRequest) -> AsyncGenerator:
        prompt_tokens = 0
        prompt = self.get_prompt(request.messages)
        prompt_tokens += self.get_token_count(prompt)
        generation_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=request.temperature >= 1e-3,
        temperature=request.temperature,
        top_p=request.top_p)
        decode_config = dict(skip_special_tokens=True, 
                             spaces_between_special_tokens=False,
                             skip_prompt=True)
        streamer = TextIteratorStreamer(self.tokenizer, **decode_config)
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        generation_kwargs = dict(input_ids, streamer=streamer,
                                generation_config=generation_config)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token


class Vicuna(LLAMA2BASE):

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


class LLAMA2(LLAMA2BASE):

    def get_prompt(self, prompts: list) -> str:
        prompt = ''
        for message in prompts:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f'[INST] <<SYS>>\n{content}\n<</SYS>>\n\n'
            elif role == 'assistant':
                prompt += f" {content}"
            else: 
                prompt += f'[INST] {content} [/INST]'
        return prompt


class NousHermes(LLAMA2BASE):

    def get_prompt(self, prompts: list) -> str:
        prompt = ''
        for message in prompts:
            role = message['role']
            content = message['content']
            if role == 'input':
                prompt += f'### Input:\n{content}\n\n'
            elif role == 'assistant':
                prompt += f"### Response:\n{content}\n\n"
            else: 
                prompt += f'### Instruction:\n{content}\n\n'
        prompt += "### Response:\n"
        return prompt


class Mistral(LLAMA2):
    def get_prompt(self, prompts: list) -> str:
        prompt = ''
        for message in prompts:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f'<s>[INST] {content} [/INST]'
            elif role == 'assistant':
                prompt += f" {content}</s>"
            else: 
                prompt += f'[INST] {content} [/INST]'
        return prompt