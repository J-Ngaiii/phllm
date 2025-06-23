from phllm.models import get_ProkBERT
MODEL_CONFIG = {
    'prokbert': get_ProkBERT
}

def get_model(llm='prokbert', rv='model'):
    llm, rv = llm.lower(), rv.lower()
    match llm:
        case 'prokbert':
            return MODEL_CONFIG.get(llm)
        case _:
            raise ValueError(f"Unkown llm {llm}")
