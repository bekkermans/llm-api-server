# LLM API Server

## Introduction
The LLM API Server is an API application designed to serve open-source  Language Models (LLMs) with full capabilities of OpenAI’s API.
This project is built upon the [Ray](https://docs.ray.io/) framework.

## Supported LLMs
The LLM API Server supports a range of language models for text encoding and text generation:
### Text encoding (embedding)
- [Sentence transformers](https://huggingface.co/sentence-transformers)
- [Instructor](https://huggingface.co/hkunlp/instructor-large)

### Text generation
- [Vicuna](https://huggingface.co/lmsys)
- [LLAMA 2](https://huggingface.co/meta-llama)

## Configuration
The LLM API Server utilizes the Ray configuration file format.

To configure the LLM API Server, modify the `args` section within the configuration file:

```yaml
args:
  llms:
    # This section configures text generation LLMs
    completions:
      - name: "/models/completions/Llama-2-13b-chat-hf" # Model weights path
        class: "llama2" # Implemented class 
        params: # (Optional) Parameters pass into the model's __init__ method
          load_in_8bit: True
          device_map: "cuda:0"
      - name: "/models/completions/vicuna-7b"
        class: "vicuna"
        params:
          torch_dtype: "auto"

    # This section configures text embedding LLMs
    embedding:
      - name: "/models/embeddings/instructor-large"
        class: "sentence-transformers"
      - name: "/models/embeddings/all-MiniLM-L6-v2"
        class: "sentence-transformers"
```
In this configuration, you have the flexibility to specify text generation LLMs (completions), text embedding LLMs (embedding), or both. 
For each type of LLM, you can define multiple models using their name, class, and additional params if needed.

## Deployment
### Run the application as a Podman local container
 1. Download the pre-trained LLM weights from the respective model repositories and save them in a folder, for example `./models`
 2. Build the container. 
    Use the provided Dockerfile to build the container image:
  ```bash
    $ podman build -t llm-api-server:latest .
  ```
3. Create a Ray configuration yaml file. There is an example configuration file `config.yaml.example`
4. Run the container. Here is an example command:
  ```bash
    $ podman run --rm -d \
    -p 7000:7000 \
    -p 8265:8265 \
    -e CONFIG_FILE_NAME=config.yaml \
    -v $(pwd)/config.yaml:/llm_api_server/config.yaml \
    -v $(pwd)/models:/models \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --security-opt=label=disable \
    --cgroup-manager=cgroupfs \
    --pids-limit=5000 \
    llm-api-server:latest
  ```
  ```
  Where:
  -p 7000:7000 - Exposes the API port
  -p 8265:8265 - Exposes the Ray Dashboard port
  -e CONFIG_FILE_NAME=config.yaml - Specifies the name of the Ray configuration file
  -v $(pwd)/config.yaml:/llm_api_server/config.yaml - Maps the Ray configuration file to the container
  -v $(pwd)/models:/models - Maps the model weights to the container
  ```
## Usage
### OpenAI python libray
More information how to use `openai` python library you can find [here](https://github.com/openai/openai-python)
```python
import openai

openai.api_base = 'http://localhost:7000/v1'
openai.api_key = ''
EMB_MODEL = '/models/embeddings/all-MiniLM-L6-v2'
COMPL_MODEL = '/models/complitions/Llama-2-13b-chat-hf'

INPUT = ['Once upon a time']

# List models
models = openai.Model.list()
print(models)

# Create embeddings
embedding = openai.Embedding.create(
        input=INPUT,
        model=EMB_MODEL)

print(embedding)

# Text generation 
text_completion = openai.Completion.create(
        model=COMPL_MODEL,
        n=1,
        temperature=0.7,
        prompt=INPUT,
        )
print(text_completion)
```