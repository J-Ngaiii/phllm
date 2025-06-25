from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
import datetime
import json
import os

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
      seq_list.append(str(r.seq))

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
    n_subdivision = 4000
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

    strain_dir = Path(strain_dir)
    strains = {}

    if seq_report:
      seq_num_list = []
      num_base_pairs = []
      max_num_base_pairs = 0
      for file_path in strain_dir.glob('*' + file_type):
          identifier = file_path.stem  # filename without extension
          print('=============', 'Parsing: ', identifier, '=============')
          seq_list = load_fna_seq(file_path, retain_seq_obj)
          seq_num_list.append(len(seq_list))
          print(f'{identifier} Num Sequences: {len(seq_list)}')

          n_base_pairs = sum([len(seq) for seq in seq_list])
          num_base_pairs.append(n_base_pairs)
          print(f'{identifier} Num Base Pairs: {n_base_pairs}')

          if n_base_pairs > max_num_base_pairs:
            max_num_base_pairs = n_base_pairs
          print('\n')

          strains[identifier] = seq_list

      print(f"Loaded {len(strains)} {strn_or_phg}s from {strain_dir}")
      print(f"Total number of sequences processed: {sum(seq_num_list)}")
      print(f"Total number of base pairs encountered: {sum(num_base_pairs)}")
      print(f"Maximum length sequence: {max_num_base_pairs}")

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
      for file_path in strain_dir.glob('*' + file_type):
          identifier = file_path.stem  # filename without extension
          seq_list = load_fna_seq(file_path, retain_seq_obj, filt)
          seq_num_list.append(len(seq_list))

          n_base_pairs = sum([len(seq) for seq in seq_list])
          num_base_pairs.append(n_base_pairs)

          if n_base_pairs > max_num_base_pairs:
            max_num_base_pairs = n_base_pairs

          strains[identifier] = seq_list

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

def rt_dicts(path = None, microbe: str = 'e_coli', strn_or_phg: str = 'strain', seq_report=False, debug=False, pad_key = False, n_subdivision = 4000):
    """For now this function simply returns a dictionary of extracted strains.
    Dictionary takes the form of keys being strain/phage names and """

    if path is None:
      path = f'/content/drive/MyDrive/phage_public_datasets/{microbe}/genomes/{strn_or_phg}_genomes/'
    assert isinstance(path, str), f"Inputted path is not a string but type {type(path)}"

    strain_dict = load_fna(path, strn_or_phg=strn_or_phg, seq_report=seq_report, debug=debug, pad_key=pad_key, n_subdivision=n_subdivision)
    return strain_dict

def by_row_embedding_saver(arr, pads_per, path, name):
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

    - pads_per : dict
        A dictionary where keys are strain names (or ids) and values are the number of padding elements for each strain.

    - path : str
        The directory path where the embeddings should be saved.

    - name : str
        The base name for the saved embeddings (e.g., `ephage_embed`).
    """
    assert len(pads_per) == arr.shape[0], f"Dimension mismatch, pads dict has {len(pads_per)} values and arr has shape {arr.shape[0]} rows."
    os.makedirs(path, exist_ok=True) # ensure path exists

    for i, (strain_name, pad_count) in enumerate(pads_per.items()): # enumerate creates an iterable returning an index and a tuple with pairs of elems from the iterable being enumerated
        valid_len = arr.shape[1] - pad_count # extract how many embeddings to keep
        valid_embedding = arr[i, :valid_len, :]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file_name = f"{name}_{strain_name}_{timestamp}.npy"
        np.save(os.path.join(path, file_name), valid_embedding)

        print(f"Saved embeddings for {name} {strain_name} at {file_name}", f"{i+1}/{len(pads_per)}")
    print(f"Finished saving {len(pads_per)} {name} embeddings!\n")

def save_to_dir(dir_path, embeddings, pads, name='ecoli', strn_or_phage='strain'):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Define filenames based on inputs
    if name == 'ecoli':
        if strn_or_phage == 'strain':
            embedding_name = "estrain_embed"
            pad_name = "estrain_pads"
        elif strn_or_phage == 'phage':
            embedding_name = "ephage_embed"
            pad_name = "ephage_pads"
        else:
            raise ValueError(f"Unknown strn_or_phage type: {strn_or_phage}")
    else:
        raise ValueError(f"Unknown name: {name}")

    # Save embeddings and padding
    np.save(os.path.join(dir_path, f'{embedding_name}_{timestamp}.npy'), embeddings)
    with open(os.path.join(dir_path, f'{pad_name}_{timestamp}.json'), 'w') as f:
        json.dump(pads, f)
   
   
    
    