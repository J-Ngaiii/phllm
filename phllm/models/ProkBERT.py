PROKBERT_AVAILABLE = None
_model = None
_tokenizer = None
_tokenizer_config = None

def _load_prokbert():
    global _model, _tokenizer, _tokenizer_config, PROKBERT_AVAILABLE
    if PROKBERT_AVAILABLE is not None:
        return

    try:
        from transformers import AutoModel, AutoTokenizer
        path = 'neuralbioinfo/prokbert-mini-long'
        _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _tokenizer_config = {
            "max_length": _tokenizer.model_max_length,   
            "padding": "max_length",
            "truncation": True
        }
        _model = AutoModel.from_pretrained(path, trust_remote_code=True)
        PROKBERT_AVAILABLE = True
    except Exception as e:
        PROKBERT_AVAILABLE = False
        _model = _tokenizer = None
        print(f"[WARN - prokbert loader] 'ProkBERT' could not be imported: {e}")
        import traceback; traceback.print_exc()

def is_ProkBERT_available():
    _load_prokbert()
    return PROKBERT_AVAILABLE

def get_ProkBERT(rv='model'):
    _load_prokbert()
    return_command = rv.lower()
    if not PROKBERT_AVAILABLE:
        print("ProkBERT not available; returning None.")
        return None
    
    if return_command == 'model':
        return _model 
    elif return_command == 'tokenizer':
        return _tokenizer
    elif return_command == 'tokenizer_config':
        return _tokenizer_config    
    else:
        raise ValueError(f"Unkown return value {rv}")
