import openai

INPUT = ["Dogs don't like cats", 
         "Cats don't like dogs", 
         "Humans like dogs and cats"]

if __name__ == '__main__':
    openai.api_base = 'http://localhost:7000/v1'
    openai.organization = ''
    openai.api_key = ''
    
    # Test list models method
    models = openai.Model.list()
    print(models)

    # Test embeddings method
    result_list = []
    for model_spec in models['data'][1:]:
        resp = openai.Embedding.create(
        input=INPUT,
        model=model_spec['id'])
        emb_len = [len(emb['embedding']) for emb in resp['data']]
        result_list.append(
            {
                "model_name": model_spec['id'],
                "embeddings": emb_len
            }
        )
    print(result_list)