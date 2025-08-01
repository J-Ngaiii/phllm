GLM2_AVAILABLE = None
_model = None
_tokenizer = None

def _load_glm2():
    global _model, _tokenizer, GLM2_AVAILABLE
    if GLM2_AVAILABLE is not None:
        return

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        _model = AutoModel.from_pretrained(
            'tattabio/gLM2_650M', 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).cuda()
        
        _tokenizer = AutoTokenizer.from_pretrained(
            'tattabio/gLM2_650M', 
            trust_remote_code=True
        )
        
        GLM2_AVAILABLE = True
    except Exception as e:
        GLM2_AVAILABLE = False
        _model = None
        _tokenizer = None
        print(f"[WARN - glm loader] 'gLM2' could not be imported: {e}")
        import traceback; traceback.print_exc()

def is_gLM2_available():
    _load_glm2()
    return GLM2_AVAILABLE

def get_gLM2(rv='model'):
    _load_glm2()
    if not GLM2_AVAILABLE:
        print("gLM2 not available; returning None.")
        return None
    return _model if rv.lower() == 'model' else _tokenizer
