from phllm.models.ProkBERT import is_ProkBERT_available, get_ProkBERT
from phllm.models.Evo2 import is_Evo2_available, get_Evo2
from phllm.models.megaDNA import is_megaDNA_available, get_megaDNA
from phllm.extract.chunkers import extract_embeddings_prokbert, extract_embeddings_evo2, extract_embeddings_megadna
from typing import Callable

MODEL_CONFIG = {
    'prokbert': {
        'operational_status': is_ProkBERT_available,
        'getter': get_ProkBERT,
        'extractor': extract_embeddings_prokbert
    },
    'evo2': {
        'operational_status': is_Evo2_available,
        'getter': get_Evo2,
        'extractor': extract_embeddings_evo2
    }, 
    'megadna': {
        'operational_status': is_megaDNA_available,
        'getter': get_megaDNA,
        'extractor': extract_embeddings_megadna
    }
}

def check_status(llm='prokbert'):
    llm = llm.lower()
    status_func = MODEL_CONFIG[llm]['operational_status']
    is_ok = status_func()
    if not is_ok:
        print(f"[WARN - Status Checker] Model '{llm}' not operational.")
    else:
        print(f"Status Checker - Model '{llm}' operational!")
    return is_ok
    
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
