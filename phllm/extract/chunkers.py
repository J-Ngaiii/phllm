import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch

import time

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
    log_path: str = "./experiment_logs", 
    test_mode=False
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

  # Setup Cuda
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  print(f"Extract_Embeddings Cuda Check:\nModel is on device: {next(model.parameters()).device}")

  if test_mode:
    print("Test mode active, only extracting 3 strain and phage .fna files")
    for i in range(3):
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
      representations = last_hidden_states.mean(axis=1) #NOTE: we perform mean pooling across tokens
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

  else:
    print("Test mode processing, printing all detected files")
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
      representations = last_hidden_states.mean(axis=1) #NOTE: we perform mean pooling across tokens
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
  batch_size, subdivisions = arr.shape
  out = np.array(embeddings) # Shape: (d, B, E), d for # of divisions, B for number of strains/phages and E for max length of an embedding
  # this happens because we're for-looping through the sub-divisions, so each element in embeddings is a 2d matrix representing the embedding representations of B strains for that particular sub-division
  out = out.transpose(1, 0, 2) # Shape: (B, d, E)
  if out.shape[0] != batch_size or out.shape[1] != subdivisions:
    print(f"First two dimensions of output should be {(batch_size, subdivisions)} but were {out.shape[:2]}.")

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

