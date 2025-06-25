from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # phllm/phllm/pipeline → phllm/

PATHS = {
    'ecoli': {
        'raw_strain': str(ROOT / 'data' / 'raw' / 'ecoli' / 'strains'),
        'raw_phage': str(ROOT / 'data' / 'raw' / 'ecoli' / 'phages'),
        'clean_strain': str(ROOT / 'data' / 'embeddings' / 'ecoli' / 'strains'),
        'clean_phage': str(ROOT / 'data' / 'embeddings' / 'ecoli' / 'phages'),
    }
}

def get_paths(bacteria='ecoli'):
    bacteria = bacteria.lower()
    bacteria_dict = PATHS.get(bacteria)
    if not bacteria_dict:
        raise ValueError(f"Unknown bacteria: {bacteria}")
    
    return (
        bacteria_dict['raw_strain'],
        bacteria_dict['raw_phage'],
        bacteria_dict['clean_strain'],
        bacteria_dict['clean_phage']
    )