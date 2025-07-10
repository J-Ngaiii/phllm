import numpy as np
import argparse

def load_npy(filename, phg_or_strn=None, path=None, allow_pickle=True) -> np.ndarray:
    config = {
        'ecoli': {
            'strain': '/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/strains',
            'phage': '/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/phages'
        }
    }

    bacterium = None
    for key in config:
        if key in filename:
            print(f"Found matching key: {key}")
            bacterium = key
    if bacterium is None:
        print(f"Unknown bacterium. Function only handles: {list(config.keys())}")
        raise ValueError()

    bacterium_paths = config[bacterium]
    
    if phg_or_strn in bacterium_paths:
        print(f"Printing: {phg_or_strn}")
        path = bacterium_paths[phg_or_strn]
    if path is None:
        for t in bacterium_paths:
            if t in filename:
                print(f"Found matching type: {t}")
                path = bacterium_paths[t]
                break
        if path is None:
            print(f"Unknown type. Function only handles: {list(bacterium_paths.keys())}")
            raise ValueError()

    full_path = f'{path.rstrip("/")}/{filename}' + ('' if filename.endswith('.npy') else '.npy')
    data = np.load(full_path, allow_pickle=allow_pickle)

    print(f"Filename: {filename}, Shape: {data.shape}\nFile:\n{data}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a npy array from local data folder.")
    parser.add_argument('--filename', default=None, help='Name of the .npy file (without full path).')
    parser.add_argument('--pos', default=None, choices=['phage', 'strain'], help='Specify whether file is from a "phage" or "strain".')
    args = parser.parse_args()

    if args.filename is None:
        # default test run
        phage = 'ecoli_NAN33_P2_2025-07-04_15-47-46'
        strain = 'ecoli_LMR3158_2025-07-10_10-57-31'

        load_npy(phage, 'phage')
        load_npy(strain, 'strain')
    else:
        load_npy(args.filename, args.pos)
