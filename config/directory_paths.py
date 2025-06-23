PATHS = {
    'ecoli' : {'raw_strain':'/data/raw/ecoli/strains', 
               'raw_phage':'/data/raw/ecoli/phages', 
               'clean_strain':'/data/embeddings/ecoli/strains', 
               'clean_phage':'/data/embeddings/ecoli/phages', }
}

def get_paths(bacteria='ecoli'):
    bacteria = bacteria.lower()
    return PATHS.get('raw_strain'), PATHS.get('raw_phage'), PATHS.get('clean_strain'), PATHS.get('clean_phage')