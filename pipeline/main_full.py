from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
import datetime
import json

from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel

import os

from phllm.extract.chunkers import complete_n_select, extract_embeddings

# *------------------
#  models
# *------------------

# using the prokbert-mini model
model_name_path = 'neuralbioinfo/prokbert-mini-long'
tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
# We are going to use base, encoder model
model = AutoModel.from_pretrained(model_name_path, trust_remote_code=True)

def get_ProkBERT(rv='model'):
    rv = rv.lower()
    if rv == 'model':
        return model
    elif rv == 'tokenizer':
        return tokenizer
    else:
        raise ValueError(f"Unkown return option {rv}")
    
# *------------------
#  Config > model_factory
# *------------------

MODEL_CONFIG = {
    'prokbert': get_ProkBERT
}

def get_model(llm='prokbert', rv='model'):
    llm, rv = llm.lower(), rv.lower()
    match llm:
        case 'prokbert':
            return MODEL_CONFIG.get(llm)
        case _:
            raise ValueError(f"Unkown llm {llm}")

# *------------------
#  Config > directory_paths
# *------------------

PATHS = {
    'ecoli' : {'raw_strain':'/data/raw/ecoli/strains', 
               'raw_phage':'/data/raw/ecoli/phages', 
               'clean_strain':'/data/embeddings/ecoli/strains', 
               'clean_phage':'/data/embeddings/ecoli/phages', }
}

def get_paths(bacteria='ecoli'):
    bacteria = bacteria.lower()
    return PATHS.get('raw_strain'), PATHS.get('raw_phage'), PATHS.get('clean_strain'), PATHS.get('clean_phage')

# *------------------
#  helpers: loading and saving
# *------------------

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

# *------------------
#  preprocessing
# *------------------

def select(d, selector, selector_args=None):
  """The ecoli_strain dicts have values as lists with elements of those lists being strings corresponding to sequences (sometimes a strain has multiple sequences).
  This function implements a selector to return a 1d list of strings, selecting a sequence for each strain/phage."""
  if selector_args is not None:
    if not isinstance(selector_args, (list, tuple)):
        selector_args = [selector_args]
    return selector(d, *selector_args)
  else:
    return selector(d)

def _first_n_check(lst, selector_args):
  assert selector_args is not None, f"Need n arg to check 'first n' selector but got None."
  assert type(selector_args) is int, f"Selector arg is an {type(selector_args)} not an int."

  for i in lst:
    if len(i) != selector_args:
      return False
  return True

def _first_selector(d):
  """Selects the first sequence from each strain and returns it as a list."""
  assert type(d) is dict, f"Inputted datastructure is not a dictionary but type {type(d)}"
  return [val[0] for val in d.values()] #d.values is a list

def _n_select_start_check(lst: list, start: int, key_obj: dict):
  """Given a key_object dictionary and a start index, this function will check that the ith value of the key_object against the ith entry in list.
  Making sure that the start-th letter of the condensed string that occupies the ith value of the key_object is the same as the first letter of that with entry in the list.

  Basically it checks that a _n_selector is properly starting at index n.

  NEED TO TEST TO SEE IF ITS WORKING PROPERLY."""

  assert start is not None, f"Need n arg to check 'first n' selector but got None."
  assert type(start) is int, f"Selector arg is an {type(start)} not an int."
  assert type(key_obj) is dict, f"Key object to check against must be a dict, currently {type(key_obj)}."
  assert len(key_obj) == len(lst), f"Elem count mismatch, {len(key_obj)} values in Key object and {len(lst)} values in inputted lst."

  for (sequences, lst_elem) in zip(key_obj.values(), lst):
    assert isinstance(sequences, list), f"Expected list of strings, got {type(sequences)}"
    assert all(isinstance(seq, str) for seq in sequences), f"Non-string sequence found in {sequences}"
    assert isinstance(lst_elem, str), f"Expected string in lst, got {type(lst_elem)}"

    complete_seq = ''.join(sequences)
    if start >= len(complete_seq): # start index is beyond the lenth of the sequence
        return False  #NOTE: we might want to change this if we expect some sequences to be below N sometimes??

    if not lst_elem and complete_seq:  # lst_elem is empty but there exists a sequence
        return False

    if complete_seq[start] != lst_elem[0]:
        return False
  return True

def _n_selector(d: dict, start: int, n: int, iterative=False) -> list:
  """Selects the first n base pairs starting from index 'start' iterating through sequences from each strain and returns it as a list.
  Can add handling for lists of Seq.IO objects later?"""
  assert isinstance(d, dict), f"Expected a dictionary, got {type(d)}"

  result = []

  if iterative:
    for key, sequences in d.items():
        assert isinstance(sequences, list), f"Expected list of strings, got {type(sequences)}"
        counter = n
        collected = ''
        current_index = 0  # total index across sequences

        for seq in sequences:
            assert isinstance(seq, str), f"Expected string in sequences, got {type(seq)}"
            seq_len = len(seq)

            if current_index + seq_len <= start:
                # skip this sequence entirely, shouldn't start yet
                current_index += seq_len
                continue

            # of current_index + seq_len > start we calculate the exact starting index
            local_start = max(0, start - current_index) #index of the current seq where we should start processing
            available = seq[local_start:]

            if len(available) <= counter:
                collected += available
                counter -= len(available)
            else:
                collected += available[:counter]
                counter = 0
                break

            current_index += seq_len

        result.append(collected)
  else:
    for key, sequences in d.items():
      assert isinstance(sequences, list), f"Expected list of strings, got {type(sequences)}"

      complete_seq = ''.join(sequences)
      relevant_seq = complete_seq[start:]
      collected = relevant_seq[:n]  # even if relevant_seq is shorter than n, that's fine
      result.append(collected)

  assert _selector_output_checker(result, 'first n', n), f"Not all elems of output list have length {n}."
  assert _selector_output_checker(result, 'n select', (start, n, d)), f"Not all elems of output list are starting at the right elem."
  return result

def _selector_output_checker(lst, selector_logic, selector_args=None):
  """Checks that a flattened list works according to some designated logic."""
  assert isinstance(lst, list), f"Input lst is a {type(lst)} not a list object."

  if selector_logic == 'first n':
    return _first_n_check(lst, selector_args)
  if selector_logic == 'n select':
    # selector_args[0] should be start, [1] should be n value and [2] should be the dictionary to check against
    return _first_n_check(lst, selector_args[1]) and _n_select_start_check(lst, selector_args[0], selector_args[2])

# *------------------
#  embedding extraction
# *------------------

def complete_n_select(d: dict, n: int, rt_array=True, debug=False) -> np.ndarray:
  """
  Subdivides sequences from a dictionary of DNA base pair strings into fixed-length sub-samples.

  Args:
  ----------
  - d : dict
      A dictionary mapping strain/phage identifiers to a list of DNA base pair sequences (strings).
      This is the standard dictionary output from the function 'rt_dicts'.
        Example:
        {
            'strain_A': ['ATCG', 'GCTA'],
            'strain_B': ['TTGG', 'CCAA']
        }

  - n : int
      The fixed length of each sub-sample (i.e., the number of base pairs per segment).

  - rt_array : bool, optional (default=True)
      If True:
          Returns a 2D NumPy array of shape (B, d), where B is the number of strains/phages
          and d is the number of n-sized sub-samples per strain (padded if necessary to align all rows).
      If False:
          Returns a dictionary mapping each strain/phage to its list of n-sized sub-samples (without padding).

  debug : bool, optional (default=False)
      If True, prints debug information such as intermediate sub-sample arrays and padding amounts.

  Returns (two outputs in the order as listed below):
  -------
  - out : np.ndarray or dict
      If rt_array is True:
          out : np.ndarray
              A 2D NumPy array of shape (B, d) where each element is a string of base pairs of length n
              (or an empty string for padded entries).
      Else:
          out : dict
              A dictionary where each key maps to a list of n-sized string sub-samples for that strain.
  - pads_per_val: dict
      A dictionary mapping each strain/phage key to the number of padded (empty string) entries added
      to ensure all rows in the output array are the same length.

  Notes:
  -----
  - The final segment in each sequence may be shorter than n if the total number of base pairs
    is not divisible by n. No further truncation is performed.
  - If `rt_array=True`, shorter rows are padded with empty strings ('') to match the longest row.
  - Padding is tracked in `pads_per_val` to allow downstream filtering if needed.
  """
  assert isinstance(d, dict), f"First arg must be a dictionary, currently {type(d)}."
  assert all(isinstance(seq_lst, list) and all(isinstance(seq, str) for seq in seq_lst) for seq_lst in d.values()), \
    "All dictionary values must be lists containing string sequences."
  assert isinstance(n, int), f"Second arg must be a int, currently {type(n)}."

  def _n_subdivide(seq: str, n: int, numpy=True):
    """Keeps dividing a string sequence of base pairs into sub-samples of size n until the sequences is completely exhausted.
    Return a list of sub-sample divisions."""
    curr = 0
    arr = []
    while curr + n < len(seq): #stop an iteration early
      arr.append(seq[curr:curr+n])
      curr += n
    arr.append(seq[curr:]) #then we can just append the rest of the sequences using curr

    if numpy:
      return np.array(arr)
    else:
      return arr

  full_seqs = [''.join(seq_lst) for seq_lst in d.values()]
  if rt_array:
    initial_sub_samples = [_n_subdivide(seq, n, False) for seq in full_seqs] # not all elems will have arrays of the same length

    if debug:
      print(f"Initial sub samples:\n {initial_sub_samples}\n")

    # padding section
    max_length = max(len(x) for x in initial_sub_samples)
    padded = [] # POTENTIAL ISSUE: need to normalize the length, but unsure if model hallucinates embeddings for empty padding inputs
    num_pads = []
    for samp in initial_sub_samples:
        assert isinstance(samp, list), f"Samples from initial sub samples is {type(samp)} but should be type list."
        discrepancy = max_length - len(samp)
        padded.append(np.array(samp + [''] * discrepancy))
        num_pads.append(discrepancy)

    if debug:
      print(f"Num pads:\n {num_pads}\n")

    pads_per_val = dict(zip(d.keys(), num_pads))
    out = np.array(padded, dtype=object).reshape((len(d), -1))
    # assert len(out.shape) == 2, f"Output has {len(out.shape)} dimension but should have 2 dimensions."
    # assert out.shape[0] == len(dict), f"Output has {out.shape[0]} rows but should {len(dict)}."
    return out, pads_per_val

  else:
    sub_samples = map(lambda seq: _n_subdivide(seq, n, False), full_seqs)
    return dict(list(zip(d.keys(), sub_samples))), pads_per_val

def extract_embeddings(
    arr: list[list],
    n: int,
    tokenize_func: callable,
    model: callable,
    out_path: str = './experiments',
    log_path: str = "./experiment_logs"
    ):

  """This function first tokenizes a dataset then then extract the embedding representations.
  Given that the inputted array is B x d columns for d subdivisions, each of size n, of the original genomes of all B strains,
  this function outputs a B x d x E matrix where each column is a embedding representation for one of the subdivisions.
  So you have B strains, d embeddings and the max length of an embedding is E values long."""


  embeddings = []
  max_embedding_dim = 0

  print("==============", "BEGINNING EMBEDDING EXTRACTION", "==============")

  times = []
  prev_time = time.time()
  start_time = prev_time

  for i in range(arr.shape[1]):
    curr = arr[:, i]
    assert all([isinstance(seq, str) for seq in curr]), f"Not all elements in inputted array are type str."

    # THIS PART IS SPECIFIC TO ProkBERT
    ds = Dataset.from_dict({"base_pairs": curr})
    tokenized = ds.map(tokenize_func, batched=True, num_proc=1)

    training_args = TrainingArguments(
    output_dir=out_path,  # Output directory
    per_device_eval_batch_size=16,  # Batch size for evaluation
    remove_unused_columns=True,  # Ensure compatibility with input format
    logging_dir=log_path,  # Logging directory
    report_to="none",  # No reporting needed
    )

    # Set up the Trainer for prediction and evaluation
    trainer = Trainer(
        model=model,  # Dummy model
        args=training_args,  # Evaluation arguments
    )
    Y_hat = trainer.predict(tokenized)
    last_hidden_states = Y_hat.predictions[0]
    representations = last_hidden_states.mean(axis=1) #NOTE:
    max_embedding_dim = max(max_embedding_dim, representations.shape[1])
    embeddings.append(representations)

    print(f"{i+1}/{arr.shape[1]} embeddings extracted.")

    elapsed = time.time() - prev_time
    times.append(elapsed)
    estimated_seconds = np.min(times) * (arr.shape[1] - (i + 1)) # empirically min does okay
    if estimated_seconds / 60 < 1:
      estimated_time = np.round(estimated_seconds, decimals=4)
      print(f"Estimated time till completion: {estimated_time} seconds.")
    elif estimated_seconds /60**2 > 1:
      estimated_time = np.round(estimated_seconds / 60**2, decimals=4)
      print(f"Estimated time till completion: {estimated_time} hours.")
    else:
      estimated_time = np.round(estimated_seconds / 60, decimals=4)
      print(f"Estimated time till completion: {estimated_time} minutes.")

  # POTENTIAL ISSUE
  out = np.array(embeddings) # Shape: (d, B, E), d for # of divisions, B for number of strains/phages and E for max length of an embedding
  # this happens because we're for-looping through the sub-divisions, so each element in embeddings is a 2d matrix representing the embedding representations of B strains for that particular sub-division
  out = out.transpose(1, 0, 2) # Shape: (B, d, E)
  assert out.shape[:2] == arr.shape, f"First two dimensions of output should be {arr.shape} but were {out.shape[:2]}."

  total_time = time.time() - start_time
  if total_time / 60 < 1:
    estimated_time = np.round(total_time, decimals=4)
    print(f"Total time taken: {estimated_time} seconds.")
  elif total_time / 60**2 > 1:
    estimated_time = np.round(total_time / 60**2, decimals=4)
    print(f"Total time taken: {estimated_time} hours.")
  else:
    estimated_time = np.round(total_time / 60, decimals=4)
    print(f"Total time taken: {estimated_time} minutes.")
  print("==============", "END OF EMBEDDING EXTRACTION", "==============")

  return out

# *------------------
#  main
# *------------------

def main():
    # Configurations
    LLM = 'prokbert'
    CONTEXT_WINDOW = 4000
    BACTERIA = 'ecoli'
    ECOLI_STRAINS, ECOLI_PHAGES, ECOLI_STRAIN_EMBEDDINGS, ECOLI_PHAGE_EMBEDDINGS = get_paths(bacteria=BACTERIA)
    
    # Pulling genomes into dictionaries to load into model
    ecoli_strains = rt_dicts(path=ECOLI_STRAINS, seq_report=True)
    num_ecoli_strains = len(ecoli_strains)
    ecoli_strains_subdivisions, ecoli_strains_pads = rt_dicts(path=ECOLI_STRAINS, pad_key=True)

    ecoli_phages = rt_dicts(path=ECOLI_PHAGES, strn_or_phg='phage', seq_report=True)
    num_ecoli_phages = len(ecoli_phages)
    ecoli_phages_subdivisions, ecoli_phages_pads = rt_dicts(path=ECOLI_PHAGES, strn_or_phg='phage', pad_key=True)

    # Setting up model
    tokenizer = get_model(llm=LLM, rv='tokenizer')
    model = get_model(llm=LLM, rv='model')

    def tokenize_func(examples, max_length=CONTEXT_WINDOW):
        # batch = examples["base_pairs"]
        # if isinstance(batch[0], list):
        #     batch = [item for sublist in batch for item in sublist]

        return tokenizer(
            examples["base_pairs"],  # input a list of multiple strings you want to tokenize from a huggingface Dataset object
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"# Set the maximum sequence length if needed
        )
    
    # Convert to huggingface datasets
    flattened_ecoli_strains = select(ecoli_strains, _n_selector, (0, 4000))
    flattened_ecoli_phages = select(ecoli_phages, _n_selector, (0, 4000))

    ds_strains = Dataset.from_dict({"base_pairs": flattened_ecoli_strains})
    ds_phages = Dataset.from_dict({"base_pairs": flattened_ecoli_phages})

    assert len(ds_phages) == num_ecoli_phages, f"Row Count Mistmatch, dataset has {len(ds_phages)} rows but there are {num_ecoli_phages} phages"
    assert len(ds_strains) == num_ecoli_strains, f"Row Count Mistmatch, dataset has {len(ds_strains)} rows but there are {num_ecoli_strains} phages"

    assert type(ds_strains[0]["base_pairs"]) is str, f"Type is {type(ds_strains[0]['base_pairs'])} not str"  # Should be str
    
    # Tokenize Data
    num_cores = os.cpu_count()
    tokenized_estrains = ds_strains.map(tokenize_func, batched=True, num_proc=1) #otherwise gets stuck try to multiprocess
    tokenized_ephages = ds_phages.map(tokenize_func, batched=True, num_proc=1)

    # Set up the Bacteria Strain Trainer 
    estrain_training_args = TrainingArguments(
        output_dir="./ecoli_strain_outputs",  # Output directory
        per_device_eval_batch_size=16,  # Batch size for evaluation
        remove_unused_columns=True,  # Ensure compatibility with input format
        logging_dir="./ecoli_strain_logs",  # Logging directory
        report_to="none",  # No reporting needed
    )

    # Extract Bacteria Strain Embeddings
    estrain_trainer = Trainer(
        model=model,  # Dummy model
        args=estrain_training_args,  # Evaluation arguments
    )
    estrain_predictions = estrain_trainer.predict(tokenized_estrains)
    estrain_last_hidden_states = estrain_predictions.predictions[0]
    estrain_representations = estrain_last_hidden_states.mean(axis=1)
    print(estrain_representations.shape)

    # Set up the Phage Trainer
    ephage_training_args = TrainingArguments(
        output_dir="./ecoli_phage_outputs",  # Output directory
        per_device_eval_batch_size=16,  # Batch size for evaluation
        remove_unused_columns=True,  # Ensure compatibility with input format
        logging_dir="./ecoli_phage_logs",  # Logging directory
        report_to="none",  # No reporting needed
    )

    # Extract Phage Embeddings
    ephage_trainer = Trainer(
        model=model,  # Dummy model
        args=ephage_training_args,  # Evaluation arguments
    )
    ephage_predictions = ephage_trainer.predict(tokenized_ephages)
    ephage_last_hidden_states = ephage_predictions.predictions[0]
    ephage_representations = ephage_last_hidden_states.mean(axis=1)
    print(ephage_representations.shape)

    # Chunking and Extracting Embeddings
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, CONTEXT_WINDOW)
    ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, CONTEXT_WINDOW)

    estrain_embed = extract_embeddings(estrain_n_select, 4000, tokenize_func, model)
    print(estrain_embed.shape)
    ephage_embed = extract_embeddings(ephage_n_select, 4000, tokenize_func, model)
    print(ephage_embed.shape)

    # Saving Embeddings to Directory
    save_to_dir(ECOLI_STRAIN_EMBEDDINGS, embeddings=estrain_embed, pads=estrain_pads, name=BACTERIA, strn_or_phage='strain')
    save_to_dir(ECOLI_PHAGE_EMBEDDINGS, embeddings=ephage_embed, pads=ephage_pads, name=BACTERIA, strn_or_phage='phage')

if __name__ == "__main__":
    main()
