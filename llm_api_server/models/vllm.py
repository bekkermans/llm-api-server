import torch
import uuid
from abc import abstractmethod
from typing import AsyncGenerator, Generator
from models.base import GenerativeLLM
from api_spec import ChatCompletionsRequest
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


class vLLM(GenerativeLLM):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        engine_config = AsyncEngineArgs(
            model=self.model_name,
            tokenizer=self.model_name,
            tokenizer_mode=kwargs.get('tokenizer_mode', 'auto'),
            revision=kwargs.get('revision', None),
            tokenizer_revision=kwargs.get('tokenizer_revision', None),
            trust_remote_code=kwargs.get('trust_remote_code', False),
            dtype=kwargs.get('dtype', 'auto'),
            max_model_len=kwargs.get('max_model_len', 1024),
            download_dir=kwargs.get('download_dir', None),
            load_format=kwargs.get('load_format', 'auto'),
            tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
            quantization=kwargs.get('quantization', None),
            enforce_eager=kwargs.get('quantization', False),
            seed=kwargs.get('seed', 0),
            disable_log_requests=kwargs.get('disable_logging', True),
            disable_log_stats=kwargs.get('disable_logging', True),
            engine_use_ray=False
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_config)
        self.device = 'cuda'

    @abstractmethod
    def get_prompt(self, prompts: list) -> str:
        pass

    def get_token_count(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    @torch.inference_mode()
    async def generate_text(self, request: ChatCompletionsRequest) -> dict:
        results = {}
        compl_list = []
        prompt_tokens = 0
        completion_tokens = 0
        prompt = self.get_prompt(request.messages)
        prompt_tokens += self.get_token_count(prompt)
        input_ids = self.tokenizer.encode(prompt)
        generation_config = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p)
        results_generator = self.model.generate(prompt=None, sampling_params=generation_config,
                                                prompt_token_ids=input_ids, 
                                                request_id=str(uuid.uuid4()))
        for _ in range(request.n):
            output_ids = None
            async for request_output in results_generator:
                output_ids = request_output
            gen_text = self.tokenizer.decode(output_ids.outputs[0].token_ids,
                                             skip_special_tokens=True,
                                             spaces_between_special_tokens=False)
            completion_tokens += self.get_token_count(gen_text)
            compl_list.append(gen_text)
        results['text'] = compl_list
        results['prompt_tokens'] = prompt_tokens
        results['completion_tokens'] = completion_tokens
        return results

    @torch.inference_mode()
    async def generate_stream(self, request: ChatCompletionsRequest) -> Generator:
        prompt_tokens = 0
        prompt = self.get_prompt(request.messages)
        input_ids = self.tokenizer.encode(prompt)
        prompt_tokens += self.get_token_count(prompt)
        generation_config =  SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p)
        stream = self.model.generate(prompt=None,
                                        sampling_params=generation_config,
                                        request_id=str(uuid.uuid4()),
                                        prompt_token_ids=input_ids)
        initial_len = -1
        async for outputs in stream:
            token = outputs.outputs[0].text[initial_len:]
            initial_len += len(token)
            yield token


class MistralFast(vLLM):
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


class LLAMA2Fast(vLLM):
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