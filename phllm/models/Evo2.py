EVO2_AVAILABLE = None
_model = None
_tokenizer = None

def _load_evo2():
    global _model, _tokenizer, EVO2_AVAILABLE
    if EVO2_AVAILABLE is not None:
        return

    try:
        from evo2 import Evo2
        _model = Evo2('evo2_7b')
        _tokenizer = _model.tokenizer
        EVO2_AVAILABLE = True
    except Exception as e:
        EVO2_AVAILABLE = False
        _model = _tokenizer = None
        print(f"[WARN - evo loader] 'Evo2' could not be imported: {e}")
        import traceback; traceback.print_exc()

def is_Evo2_available():
    _load_evo2()
    return EVO2_AVAILABLE

def get_Evo2(rv='model'):
    _load_evo2()
    if not EVO2_AVAILABLE:
        print("Evo2 not available; returning None.")
        return None
    return _model if rv.lower() == 'model' else _tokenizer
