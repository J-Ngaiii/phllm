from transformers import AutoTokenizer, AutoModel 

# using the prokbert-mini model
model_name_path = 'neuralbioinfo/prokbert-mini-long'
tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
# We are going to use base, encoder model
model = AutoModel.from_pretrained(model_name_path, trust_remote_code=True)

def get_ProkBERT(rv='model'):
    rv = rv.lower()
    if rv == 'model':
        return model
    elif rv == 'tokenizer':
        return tokenizer
    else:
        raise ValueError(f"Unkown return option {rv}")