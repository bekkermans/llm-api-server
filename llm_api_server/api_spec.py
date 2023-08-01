from typing import Optional, List, Dict, Union, Any
from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    model_name: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[str]]
    user: Optional[str] = None


class CompletionsRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 100000
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict] = None
    user: Optional[str] = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    functions: Optional[List[Dict]] = None
    function_call: Optional[Union[str, Any]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Union[str, List[str]] = None
    max_tokens: Optional[int] = 100000
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict] = None
    user: Optional[str] = None