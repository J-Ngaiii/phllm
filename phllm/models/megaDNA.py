_model = _tokenizer = None
MEGADNA_AVAILABLE = False

def _load_megadna():
    global _model, MEGADNA_AVAILABLE
    if MEGADNA_AVAILABLE is not None:
        return

    try:
        import torch
        device = 'cuda0' 
        model_path = "megaDNA_phage_145M.pt"
        _model = torch.load(model_path, map_location=torch.device(device))
        MEGADNA_AVAILABLE = True
    except Exception as e:
        MEGADNA_AVAILABLE = False
        _model = None
        print(f"[WARN - megadna loader] 'megaDNA' could not be imported: {e}")
        import traceback; traceback.print_exc()

def is_megaDNA_available():
    _load_megadna()
    return MEGADNA_AVAILABLE

def get_megaDNA(rv='model'):
    _load_megadna()
    if not MEGADNA_AVAILABLE:
        print("megaDBA not available; returning None.")
        return None
    return _model if rv.lower() == 'model' else _tokenizer