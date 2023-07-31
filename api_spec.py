from typing import Optional, List, Union
from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    model_name: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[str]]
    user: Optional[str] = None
