from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # phllm/phllm/pipeline → phllm/

BACT = {
    'ecoli': {
        'paths' : {
            'raw_strain': str(ROOT / 'data' / 'raw' / 'ecoli' / 'strains'),
            'raw_phage': str(ROOT / 'data' / 'raw' / 'ecoli' / 'phages'),
            'clean_strain': str(ROOT / 'data' / 'embeddings' / 'ecoli' / 'strains'),
            'clean_phage': str(ROOT / 'data' / 'embeddings' / 'ecoli' / 'phages'),
        }, 
        'names' : {
            'embedding_file' : {
                'strain' : 'estrain_embed', 
                'phage' : 'ephage_embed'
                }, 
            'padding_file' : {
                'strain' : 'estrain_pads', 
                'phage' : 'ephage_pads_'
                }, 
        }
    }
}

def get_paths(bacteria='ecoli', path=None):
    path = path.lower()
    bacteria = bacteria.lower()

    paths_dict = BACT.get(bacteria).get('paths')
    if not paths_dict:
        raise ValueError(f"Unknown bacteria: {bacteria}")
    
    if path is None:
        return (
            paths_dict['raw_strain'],
            paths_dict['raw_phage'],
            paths_dict['clean_strain'],
            paths_dict['clean_phage']
        )
    elif path in paths_dict:
        return paths_dict[path]
    
def get_filenames(bacteria='ecoli', embed_or_pad='embedding_file', strn_or_phage='strain'):
    strn_or_phage = strn_or_phage.lower()
    bacteria = bacteria.lower()
    assert bacteria in BACT, f"Unknown bacteria: {bacteria}"

    name = BACT.get(bacteria).get('names').get(embed_or_pad).get(strn_or_phage, "")
    if not name:
        raise ValueError(f"Unknown designation: {strn_or_phage}, pick either strain or phage.")
    return name