import openai
import time

INPUT = ["Dogs don't like cats", 
         "Cats don't like dogs", 
         "Humans like dogs and cats"]

CHAT_MESSAGES=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Sergey! What is your name?"},
  ]


# Test list all models method
def model_list():
    models = openai.Model.list()
    print(models)
    return models

# Test embeddings method
def embeddings(model_name, input):
    resp = openai.Embedding.create(
        input=input,
        model=model_name)
    emb_len = len(resp['data'][0]['embedding'])
    res = {
        "model_name": model_name,
        "embeddings_size": emb_len
        }
    print(res)

# Test Complition method
def complition(model_name, input, n=1, stream=False):
    if not stream:
        completion = openai.Completion.create(
        model=model_name,
        n=n,
        prompt=input,
        )
        print(completion)
    else:
        start_time = time.time()
        completion = openai.Completion.create(
        model=model_name,
        n=n,
        prompt=input,
        stream=True
        )
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in completion:
            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['text']  # extract the message
            collected_messages.append(chunk_message)  # save the message
            print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# Test ChatComplition method
def chat_complition(model_name, input, n=1, stream=False):
    if not stream:
        completion = openai.ChatCompletion.create(
        model=model_name,
        n=n,
        messages=input,
        )
        print(completion)
    else:
        start_time = time.time()
        completion = openai.ChatCompletion.create(
        model=model_name,
        n=n,
        messages=input,
        stream=True
        )
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in completion:
            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            collected_messages.append(chunk_message)  # save the message
            print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

        # print the time delay and text received
        print(f"Full response received {chunk_time:.2f} seconds after request")
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        print(f"Full conversation received: {full_reply_content}")


if __name__ == '__main__':
    openai.api_base = 'http://localhost:7000/v1'
    openai.organization = ''
    openai.api_key = ''
    
    embedding_models = []
    complition_models = []

    model_list = model_list()

    for model in model_list['data']:
        if 'complitions' in model['id'].split('/'):
            complition_models.append(model['id'])
        else:
            embedding_models.append(model['id'])
    
    for model in embedding_models:
        embeddings(model, INPUT)
    
    for model in complition_models:
        complition(model, INPUT[0])
        complition(model, INPUT)
        complition(model, INPUT, stream=True)
        chat_complition(model, CHAT_MESSAGES)
        chat_complition(model, CHAT_MESSAGES, stream=True)
