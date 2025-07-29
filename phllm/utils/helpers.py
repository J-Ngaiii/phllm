from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
from datetime import datetime # change
import json
import os

from phllm.config.config import get_filenames

def load_fna_seq(file_path: str, retain_seq_obj: bool = False, filt = None):
  """
  Returns list of sequences for a given .fna file.

  Args:
  - file_path (str): path to a SINGLE .fna file
  - retain_seq_obj (bool): signifies whether or not to keep sequences as Biopython Seq objects or convert to strings
  - filt (function): filter function that applies a filter on what Seq objects are kept when constructing the dictionary

  Return Value:
  - seq_list (list): list object containing all sequences in a particular .fna file
  """
  file_path = str(file_path)
  if not file_path.endswith('.fna'):
        raise ValueError(f"File path does not end in a .fna file. File path was: {file_path}")

  #Jonathan's Notes:
  # - SeqIO.parse(file_path, 'fasta') --> returns a iterable containing SeqRecord objects, different SeqRecord objects encode differents parts of a straing or phage genome
  # - SeqRecord: an object with a .id, .description and .seq attribute. The .seq attribute lists out a single DNA/RNA sequence exactly

  seq_list = []
  if retain_seq_obj: #if condition on the outside so we only check it once
    for r in SeqIO.parse(file_path, 'fasta'): #could wrap in lambda function to make it better looking
      seq_list.append(r.seq)
  else:
    for r in SeqIO.parse(file_path, 'fasta'):
      seq_list.append(str(r.seq)) # maintain split at this level

  if filt is not None:
    seq_list = filt(seq_list)

  return seq_list

def load_fna(
    strain_dir: str,
    file_type: str = '.fna',
    strn_or_phg: str = 'strain',
    retain_seq_obj: bool = False,
    seq_report: bool = False,
    filt = None,
    debug = False,
    pad_key = False,
    plots = False, 
    n_subdivision = 4000, 
    test_mode = False, 
    test_count = 3
    ) -> Dict[str, np.ndarray]:
    """
    Load all sequences from a directory (.'file_type' files, default is .fna).

    Args:
    - strain_dir (str): path to a folder containing .fna files
    - retain_seq_obj (bool): tells 'load_fna_seq' function whether or not to keep sequences as Biopython Seq objects or convert to strings
    - seq_report (bool): designate whether or not to print the number of sequence objects processed per strain
    - filt (function): filter function to pass into 'load_fna_seq' that applies a filter on what Seq objects are kept
    - debug (bool): designate whether or not to return certain outputs messages
    - pad_key (bool): designates whether or not to return an answer key for the number of subdivisions and the number of pads each strain/phage should have.
    Padding answer key comes in the form of numpy arrays, first array is the number of subdivisions per strain/phage, second array is the number of pads per strain/phage.

    Return Value:
    - strains (dict): dictionary with keys being the file names and values being a list of all sequences
    """
    if debug:
      assert isinstance(debug, str), f"If debug is not false it must be a string specifying debugging output."
      assert debug in ['seq_num_list', 'num_base_pairs'], f"Debug mode asked to return {debug}, but can only return 'seq_num_list' or 'num_base_pairs'."

    if test_mode:
        print(f"Function in test mode, terminating after finding {test_count} or less {file_type} files.")
    
    strain_dir = Path(strain_dir)
    if not strain_dir.exists():
        raise FileNotFoundError(f"Directory {strain_dir} does not exist.")
    if not any(strain_dir.glob('*' + file_type)):
        print(f"Warning: No files with extension '{file_type}' found in {strain_dir}")
    strains = {}

    if seq_report:
        seq_num_list = []
        num_base_pairs = []
        max_num_base_pairs = 0
        count = 0
        for file_path in sorted(strain_dir.glob('*' + file_type)):
            identifier = file_path.stem  # filename without extension
            print('=============', 'Parsing: ', identifier, '=============')
            seq_list = load_fna_seq(file_path, retain_seq_obj, filt)
            seq_num_list.append(len(seq_list))
            print(f'{identifier} Num Sequences: {len(seq_list)}')

            n_base_pairs = sum([len(seq) for seq in seq_list])
            num_base_pairs.append(n_base_pairs)
            print(f'{identifier} Num Base Pairs: {n_base_pairs}')

            if n_base_pairs > max_num_base_pairs:
                max_num_base_pairs = n_base_pairs
            print('\n')

            strains[identifier] = seq_list # each elem is a contig

            count += 1
            if test_mode and count == test_count:
                print(f'rt_dicts test mode active: {test_count} files founds, testing complete!')
                break
            


        print(f"Loaded {len(strains)} {strn_or_phg}s from {strain_dir}")
        print(f"Total number of sequences processed: {sum(seq_num_list)}")
        print(f"Total number of base pairs encountered: {sum(num_base_pairs)}")
        print(f"Maximum length sequence: {max_num_base_pairs}")

        if plots:
            plt.hist(num_base_pairs)
            plt.title("Distribution of Sequence Lengths")
            plt.xlabel("Sequence Length")
            plt.ylabel("Frequency")
            plt.show()

        if debug == 'seq_num_list':
            print(f"Returning 'seq_num_list'")
            return seq_num_list
        elif debug == 'num_base_pairs':
            print(f"Returning 'num_base_pairs'")
            return num_base_pairs

        if pad_key:
            max_subdivisions = np.ceil(max_num_base_pairs / n_subdivision)
            num_subdivision = np.ceil(np.array(num_base_pairs) / n_subdivision)
            num_pads = max_num_base_pairs - num_subdivision
            return num_subdivision, num_pads
    else:
        seq_num_list = []
        num_base_pairs = []
        max_num_base_pairs = 0
        count = 0
        for file_path in sorted(strain_dir.glob('*' + file_type)):
            identifier = file_path.stem  # filename without extension
            seq_list = load_fna_seq(file_path, retain_seq_obj, filt)
            seq_num_list.append(len(seq_list))

            n_base_pairs = sum([len(seq) for seq in seq_list])
            num_base_pairs.append(n_base_pairs)

            if n_base_pairs > max_num_base_pairs:
                max_num_base_pairs = n_base_pairs

            strains[identifier] = seq_list

            count += 1
            if test_mode and count == test_count:
                print(f'rt_dicts test mode active: {test_count} files founds, testing complete!')
                break

        if debug == 'seq_num_list':
            print(f"Returning 'seq_num_list'")
            return seq_num_list
        elif debug == 'num_base_pairs':
            print(f"Returning 'num_base_pairs'")
            return num_base_pairs

        if pad_key:
            max_subdivisions = np.ceil(max_num_base_pairs / n_subdivision)
            num_subdivision = np.ceil(np.array(num_base_pairs) / n_subdivision)
            num_pads = max_subdivisions - num_subdivision
            return num_subdivision, num_pads

        print(f"Loaded {len(strains)} {strn_or_phg}s from {strain_dir}")

    return strains

def rt_dicts(path = None, microbe: str = 'e_coli', strn_or_phg: str = 'strain', seq_report=False, debug=False, pad_key = False, n_subdivision = 4000, test_mode=False, test_count=3):
    """
    For now this function simply returns a dictionary of extracted strains.
    Dictionary takes the form of keys being strain/phage names.

    Full workloop is: 
    - load_fna_seq: takes in a file path --> parses an iterable of seq objects (ech containing a string of base pairs) --> returns an list of plain strings representing base pairs, one per seq object
    - load_fna: iterates through all files in a given directory path --> collects the list of plain strings representing base pairs --> returns a dictionary mapping id (eg. 370D) to this list of base pairs as strings
    - rt_dicts: returns the dictionary load_fna returns; exists as a wrapper to precondition certain settings

    - complete_n_select then takes over using the dictionary mapping ids to list of base pairs as strings to divide everything into equal subdivisions of a length compliant with compliant with the context window.
    """

    if path is None:
      path = f'/content/drive/MyDrive/phage_public_datasets/{microbe}/genomes/{strn_or_phg}_genomes/'
    assert isinstance(path, str), f"Inputted path is not a string but type {type(path)}"

    strain_dict = load_fna(path, strn_or_phg=strn_or_phg, seq_report=seq_report, debug=debug, pad_key=pad_key, n_subdivision=n_subdivision, test_mode=test_mode, test_count=test_count)
    return strain_dict

def by_row_embedding_saver(arr, pad_indices, path, name, strn_or_phg='strain', debug=False):
    """
    Takes in a 3D numpy array of embeddings and a dictionary of the number of padding values per row
    represented in each value, then eliminates invalid embeddings and saves them in a designated directory.

    Args:
    ----------
    - arr : np.ndarray
        A 3D numpy array of shape (B, d, E), where:
        - B is the number of strains/phages
        - d is the number of subdivisions (some of which may be padded)
        - E is the embedding dimension for each subdivision

    - pad_indices : dict
        Mapping from strain/phage ID to index where padding starts (i.e., number of *valid* entries).

    - path : str
        The directory path where the embeddings should be saved.

    - name : str
        The base name for the saved embeddings (e.g., `ephage_embed`).
    """
    assert len(pad_indices) == arr.shape[0], f"Dimension mismatch, pads dict has {len(pad_indices)} values and arr has shape {arr.shape[0]} rows."
    os.makedirs(path, exist_ok=True) # ensure path exists
    if debug:
        print("[DEBUG] Entered by_row_embedding_saver")

    for i, (id, pad_index) in enumerate(pad_indices.items()): # enumerate creates an iterable returning an index and a tuple with pairs of elems from the iterable being enumerated
        # handles for test mode where we only extract 3 subdivisions but the whole embedding has a lot of pads.
        # happens cuz in test mode: rt_dicts extracts all basepairs for 3 strains --> complete_n_select splits those lists of all basepairs into chunks => theres padding left over
        valid_embedding = arr[i, :pad_index, :]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Iteration {i}, timestamp {timestamp}, embedding dimension {valid_embedding.shape}")

        file_name = f"{name}_{strn_or_phg}_{id}_{timestamp}.npy"
        np.save(os.path.join(path, file_name), valid_embedding)

        print(f"Saved embeddings for {name} {strn_or_phg} {id} at {file_name}", f"{i+1}/{len(pad_indices)}")
        if debug:
            print(f"Embedding as numpy array:\n{valid_embedding}")
    print(f"Finished saving {len(pad_indices)} {name} embeddings!\n")

def save_to_dir(dir_path, embeddings, pads, pad_indices, name='ecoli', strn_or_phage='strain', full_save=False, debug=False, test_mode=False, test_count=3):
    name = name.lower()
    strn_or_phage = strn_or_phage.lower()
    
    if debug:
        print("[DEBUG] Entered save_to_dir()")
        print(f"[DEBUG] name: {name}, strn_or_phage: {strn_or_phage}, path: {dir_path}")
   
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    except Exception as e:
        print(f"[ERROR] Failed to parse datetime. Using fallback. Exception: {e}")
        timestamp = 'NULL'

    if full_save:
        print("Beginning saving process... (mode: full save)")
        # Define filenames based on inputs
        try:
            embedding_name = get_filenames(bacteria=name, embed_or_pad='embedding_file', strn_or_phage=strn_or_phage)
            pad_name = get_filenames(bacteria=name, embed_or_pad='padding_file', strn_or_phage=strn_or_phage)
            pad_indices_name = get_filenames(bacteria=name, embed_or_pad='padding_index_file', strn_or_phage=strn_or_phage)
        except ValueError as e:
            print("Failed to get names:", e)

        # Save embeddings and padding
        print(f"Saving file '{embedding_name}_{timestamp}.npy' to directory:{dir_path}")
        np.save(os.path.join(dir_path, f'{embedding_name}_{timestamp}.npy'), embeddings)
        try:
            with open(os.path.join(dir_path, f'{pad_name}_{timestamp}.json'), 'w') as f:
                json.dump(pads, f)
            with open(os.path.join(dir_path, f'{pad_indices_name}_{timestamp}.json'), 'w') as f:
                json.dump(pad_indices, f)
        except Exception as e:
            print("Failed to write JSON:", e)
            print("Pads looks like:", type(pads), list(pads)[:3])
            print("Pads indices looks like:", type(pad_indices), list(pad_indices)[:3])
    else:
        print("Beginning saving process... (mode: by-row saving)")
        by_row_embedding_saver(arr=embeddings, pad_indices=pad_indices, path=dir_path, name=name, strn_or_phg=strn_or_phage, debug=debug)
    
    
   
    
    