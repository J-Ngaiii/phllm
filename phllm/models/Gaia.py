GAIA_AVAILABLE = None
_model = None
_tokenizer = None

def _load_gaia():
    global _model, _tokenizer, GAIA_AVAILABLE
    if GAIA_AVAILABLE is not None:
        return

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        _model = AutoModel.from_pretrained(
            'tattabio/gLM2_650M_embed', 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).cuda()
        
        _tokenizer = AutoTokenizer.from_pretrained(
            'tattabio/gLM2_650M_embed', 
            trust_remote_code=True
        )
        
        GAIA_AVAILABLE = True
    except Exception as e:
        GAIA_AVAILABLE = False
        _model = None
        _tokenizer = None
        print(f"[WARN - gaia (glm2) loader] 'Gaia (gLM2)' could not be imported: {e}")
        import traceback; traceback.print_exc()

def is_Gaia_available():
    _load_gaia()
    return GAIA_AVAILABLE

def get_Gaia(rv='model'):
    _load_gaia()
    if not GAIA_AVAILABLE:
        print("Gaia (gLM2) not available; returning None.")
        return None
    return _model if rv.lower() == 'model' else _tokenizer
