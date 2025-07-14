import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')

sequence = 'ACGT'
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

layer_name = 'blocks.28.mlp.l3'

outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

print('Embeddings shape: ', embeddings[layer_name].shape)

def get_Evo2(rv='model'):
    rv = rv.lower()
    if rv == 'model':
        return model
    elif rv == 'tokenizer':
        return evo2_model.tokenizer
    else:
        raise ValueError(f"Unkown return option {rv}")