import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')
evo2_tokenizer = evo2_model.tokenizer

def get_Evo2(rv='model'):
    rv = rv.lower()
    if rv == 'model':
        return evo2_model
    elif rv == 'tokenizer':
        return evo2_tokenizer
    else:
        raise ValueError(f"Unkown return option {rv}")