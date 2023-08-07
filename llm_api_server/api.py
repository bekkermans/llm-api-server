import logging
import time
import uuid
import json
from starlette.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI

from models.vicuna import Vicuna
from models.sentence_llm import SentenceLLM
from api_spec import (
    EmbeddingsRequest,
    CompletionsRequest,
    CompletionsMessage,
    ChatCompletionsRequest,
    EmbeddingsData,
    EmbeddingsResponse,
    UsageInfo,
    ModelsResponse)

from ray import serve

logger = logging.getLogger("ray.serve")
api_app = FastAPI()

MODEL_CLASS_MAPPING = {
    "vicuna": Vicuna,
    "sentence-transformers": SentenceLLM
}


@serve.deployment(num_replicas=1)
@serve.ingress(api_app)
class API:
    def __init__(self, models_dict: dict) -> None:
        self.embedding_models_obj_dict = {}
        self.completions_models_obj_dict = {}
        self.models_card_list = []
        self.load_models(models_dict)

    @api_app.post('/v1/completions')
    async def completions(self, request: CompletionsRequest) -> JSONResponse:
        pass

    @api_app.post('/v1/chat/completions')
    async def chatcompletions(self, request: ChatCompletionsRequest) -> JSONResponse:
        model_name = request.model
        compl_model = self.get_model_by_name(model_name, 'completions')
        if compl_model == None:
            return JSONResponse({"detail": "Model not Found"}, status_code=400)
        else:
            chat_id = "chatcmpl-" + str(uuid.uuid4()).split('-')[0]
            if request.stream == False:
                resp = await compl_model.generate_text(request)
                choises_list = []
                for idx, text in enumerate(resp['text']):
                    choises_list.append({
                        "index": idx,
                        "message": CompletionsMessage(role="assistant",
                                                      content=text).dict(),
                        "finish_reason": "stop"
                    })
                cur_time = int(time.time())
                total_tokens = resp['prompt_tokens'] + resp['completion_tokens']
                return JSONResponse({
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": cur_time,
                    "choices": choises_list,
                    "usage": UsageInfo(prompt_tokens=resp['prompt_tokens'],
                                       completion_tokens=resp['completion_tokens'],
                                       total_tokens=total_tokens).dict()
                })
            else:
                result = self.stream_generation(request, compl_model, chat_id)
                return StreamingResponse(result,
                                         media_type="text/event-stream")

    @api_app.post('/v1/embeddings')
    @api_app.post('/v1/engines/{model_name}/embeddings')
    async def embeddings(self, request: EmbeddingsRequest) -> JSONResponse:
        if request.model != None:
            model = request.model
        elif request.model_name != None:
            model = request.model_name
        elif request.engine != None:
            model = request.engine
        else:
            model = ''
        if isinstance(request.input, str):
            request.input = [request.input]
        emb_model = self.get_model_by_name(model, 'embeddings')
        if emb_model == None:
            ret = JSONResponse({"detail": "Model not Found"}, status_code=400)
        else:
            tokens_count = emb_model.get_token_count(request.input)
            emb = await emb_model.encode(request.input)
            data_list = []
            for i, emb_data in enumerate(emb):
                data_list.append(
                    EmbeddingsData(embedding=emb_data,
                                   index=i).dict())
            usage_info = UsageInfo(prompt_tokens=tokens_count, 
                                   total_tokens=tokens_count).dict()
            ret = JSONResponse(EmbeddingsResponse(data=data_list, 
                                                  model=emb_model.model_name,
                                                  usage=usage_info).dict())
        return ret

    @api_app.get('/v1/models')
    async def list_models(self) -> JSONResponse:
        return JSONResponse(
            ModelsResponse(data=self.models_card_list).dict()
        )

    def stream_generation(self, request, compl_model, chat_id):
        model_name = request.model
        for k in range(request.n):
            stream_gen = compl_model.generate_stream(request)
            for i, token in enumerate(stream_gen):
                if i == 0:
                    delta = {"role": "assistant"}
                else:
                    delta = {"content": token}
                choises_list = [{
                        "index": k,
                        "delta": delta,
                        "finish_reason": "None"
                        }]
                cur_time = int(time.time())
                res = {
                        "id": chat_id,
                        "created": cur_time,
                        "object": "chat.completion.chunk",
                        "choices": choises_list,
                        "model": model_name,
                    }
                res = json.dumps(res, ensure_ascii=False)
                yield f"data: {res}\n\n"
        yield "data: [DONE]\n\n"

    def load_models(self, models_dict: dict) -> None:
        for model_tasks, models_list in models_dict.items():
            for model_params in models_list:
                model_name = model_params['name']
                if model_name != '':
                    model_class = MODEL_CLASS_MAPPING.get(model_params['class'])
                    model = model_class(model_name)
                    if model_tasks == 'completions':
                        self.completions_models_obj_dict[model_name] = model
                    elif model_tasks == 'embedding':
                        self.embedding_models_obj_dict[model_name] = model
                    model_device = model.device
                    self.models_card_list.append({
                        "id": model_name,
                        "object": "model",
                        "owned_by": "Open Source",
                        "permission": "Public"
                    })
                    logger.info(f'The model {model_name} has been '\
                                f'successfully loaded on device "{model_device}"')

    def get_model_by_name(self, model_name:str, model_task: str) -> object:
        if model_task == 'embeddings':
            ret =  self.embedding_models_obj_dict.get(model_name)
        elif model_task == 'completions':
            ret = self.completions_models_obj_dict.get(model_name)
        return ret


def app_builder(args: dict) -> API:
    return API.bind(args["llms"])