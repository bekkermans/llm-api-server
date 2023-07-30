import logging
from models import Generative, Embeddings
from starlette.responses import JSONResponse
from fastapi import FastAPI

from ray import serve

logger = logging.getLogger("ray.serve")
api_app = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(api_app)
class API:
    def __init__(self, models_dict: dict) -> None:
        self.embedding_models_obj_dict = {}
        self.completions_models_obj_dict = {}
        self.models_card_list = []
        self.load_models(models_dict)

    def load_models(self, models_dict: dict) -> None:
        for model_tasks, models_list in models_dict.items():
            for model_params in models_list:
                model_name = model_params['name']
                if model_name != '':
                    if model_params['class'] == 'llama':
                        model = Generative(model_name)
                    elif (model_params['class'] == 'instructor') or \
                        (model_name == 'sentence-transformers'):
                        model = Embeddings(model_name)
                    if model_tasks == 'completions':
                        self.completions_models_obj_dict[model_name] = model
                    elif model_tasks == 'embedding':
                        self.embedding_models_obj_dict[model_name] = model
                    model_device = model.device
                    self.models_card_list.append({
                        "id": model_name,
                        "object": "model",
                        "owned_by": "Open Source",
                        "permission": "All"
                    })
                    logger.info(f'The model {model_name} has been \
                        successfully loaded on device "{model_device}"')

    @api_app.post('/v1/completions')
    async def completions(self, model: str, prompt: str):
        model = self.get_model_by_name(model, 'completions')
        return  model.generate_text(prompt)

    @api_app.post('/v1/embeddings')
    async def embeddings(self, model: str, input: str, engine: str = '', user: str = ''):
        # TODO: batch input
        emb_model = self.get_model_by_name(model, 'embeddings')
        if emb_model == None:
            ret = JSONResponse({"detail": "Not Found"}, status_code=400)
        else:
            tokens_count = emb_model.get_token_count(input)
            logger.debug(f'Got request {input}')
            emb = await emb_model.encode(input)
            ret = JSONResponse({
                "object": "list",
                "data":[
                    {
                        "object": "embedding",
                        "embedding": emb,
                        "index": 0
                    }
                ],
                "model": emb_model.model_name,
                "usage": {
                    "prompt_tokens": tokens_count,
                    "total_tokens": tokens_count
                }
            })
        return ret

    @api_app.get('/v1/models')
    async def list_models(self):
        response = JSONResponse({
            "data": self.models_card_list,
            "object": "list"
        })
        return response

    def get_model_by_name(self, model_name:str, model_task: str):
        if model_task == 'embeddings':
            ret =  self.embedding_models_obj_dict.get(model_name)
        elif model_task == 'completions':
            ret = self.completions_models_obj_dict.get(model_name)
        return ret


def app_builder(args: dict) -> API:
    return API.bind(args["llms"])