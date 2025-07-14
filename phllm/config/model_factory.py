from phllm.models import get_ProkBERT, get_Evo2
from phllm.extract.chunkers import extract_embeddings_evo2, extract_embeddings_prokbert

from typing import Callable
MODEL_CONFIG = {
    'prokbert': {'getter' : get_ProkBERT, 'extractor' : extract_embeddings_prokbert}, 
    'evo2' : {'getter' : get_Evo2, 'extractor' : extract_embeddings_evo2}
}

def get_model(llm='prokbert', rv='model') -> Callable:
    llm, rv = llm.lower(), rv.lower()
    try:
        if llm not in MODEL_CONFIG:
            raise ValueError(f"Unkown llm {llm}")
        else:
            return MODEL_CONFIG.get(llm).get('getter')(rv=rv)
    except ValueError as e:
        raise e

def get_embedding_extractor(llm='prokbert') -> Callable:
    llm = llm.lower()
    try:
        if llm not in MODEL_CONFIG:
            raise ValueError(f"Unkown llm {llm}")
        else:
            return MODEL_CONFIG.get(llm).get('extractor')
    except ValueError as e:
        raise e
