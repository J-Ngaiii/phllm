import numpy as np
from typing import Tuple, Dict, Union
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch
import random
import time

def altered_n_select(
    d: dict,
    n: int,
    overlap_proportion: float,
    rand_score: float = 0.5,
    rt_array: bool = True,
    debug: bool = False
) -> Tuple[np.ndarray | Dict[str, list[str]], Dict[str, int], Dict[str, int]]:
  """
  Subdivides sequences from a dictionary of DNA base pair strings into fixed-length sub-samples PER CONTIG.

  Args:
  ----------
  - d : dict
      A dictionary mapping strain/phage identifiers to a list of DNA contig sequences (strings).
      This is the standard dictionary output from the function 'rt_dicts'.
        Example:
        {
            'strain_A': ['ATCG', 'GCTA'],
            'strain_B': ['TTGG', 'CCAA']
        }

  - n : int
      The fixed length of each sub-sample (i.e., the number of base pairs per segment).

  - overlap_proportion: float
      Determines the proportion of overlap between subdivided portions within any arbitrary contig

  - rand_score: float
      Determines the degree of randomness in what base pairs overlap if overlap_proportion > 0.

  - rt_array : bool, optional (default=True)
      If True:
          Returns a 2D NumPy array of shape (B, d), where B is the number of strains/phages
          and d is the number of n-sized sub-samples per strain (padded if necessary to align all rows).
      If False:
          Returns a dictionary mapping each strain/phage to its list of n-sized sub-samples (without padding).

  debug : bool, optional (default=False)
      If True, prints debug information such as intermediate sub-sample arrays and padding amounts.

  Mechanism: retains divisions between individual contigs. rather than joining all contigs together then dividing evenly in chunks of context size n,
  this function treats contigs as a 'natural division', processing contig by contig only dividing when a contig exceeds the inputted context window.
  
  Returns (two outputs in the order as listed below):
  -------
  - out : np.ndarray or dict
      If rt_array is True:
          out : np.ndarray
              A 2D NumPy array of shape (B, d) where each element is a string of base pairs up to length n
              (or an empty string for padded entries). Eachof the d strings may represent a whole contig (if it's less than the context window) 
              of subdivisions of a contig. 
      Else:
          out : dict
              A dictionary where each key maps to a list of n-sized string sub-samples for that strain.
  - pads_per_val: dict
      A dictionary mapping each strain/phage key to the number of padded (empty string) entries added
      to ensure all rows in the output array are the same length.
  - pad_starts : Dict[str, int],
      A mapping from each strain/phage key to the index where padding begins in the output array.
  """
  def get_chunks(seq: str, n: int, overlap_proportion: float, rand_score: float) -> list[str]:
        chunks = []
        if len(seq) <= n:
            return [seq]

        step_base = n * (1 - overlap_proportion)
        i = 0
        while i + n <= len(seq):
            chunks.append(seq[i:i + n])

            # Add randomness to overlap
            if 0 < overlap_proportion < 1:
                # Random jitter in [-0.5, 0.5] * step_base * rand_score
                jitter = int((random.random() - 0.5) * step_base * rand_score * 2)
                step = max(1, int(step_base + jitter))
            else:
                step = n

            i += step

        # Handle any remaining tail
        if i < len(seq) and len(seq[i:]) > n // 2:
            chunks.append(seq[-n:])

        return chunks

  # Main logic
  out_dict = {}
  max_len = 0
  pads_per_val = {}
  pad_starts = {}

  for key, contigs in d.items():
      subchunks = []
      for contig in contigs:
          contig_chunks = get_chunks(contig, n, overlap_proportion, rand_score)
          subchunks.extend(contig_chunks)
      out_dict[key] = subchunks
      if debug:
          print(f"{key}: {len(subchunks)} chunks")
      max_len = max(max_len, len(subchunks))

  if not rt_array:
      return out_dict, {}, {}

  # Pad to uniform length
  out_arr = []
  for key in d:
      row = out_dict[key]
      pad_amt = max_len - len(row)
      pads_per_val[key] = pad_amt
      pad_starts[key] = len(row)
      padded_row = row + [''] * pad_amt
      out_arr.append(padded_row)

  out_arr = np.array(out_arr, dtype=object)
  return out_arr, pads_per_val, pad_starts

def complete_n_select(d: dict, n: int, rt_array=True, debug=False) -> Tuple[np.ndarray | Dict, Dict]:
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
  - pad_starts : Dict[str, int],
      A mapping from each strain/phage key to the index where padding begins in the output array.

  Notes:
  -----
  - Padding dictionary is invalid for return dictionary mode and pad_start instead displays where the padding would've started.
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

  full_seqs = [''.join(seq_lst) for seq_lst in d.values()] # creates list where index i is the ith strain and the elem is all basepairs joined into one string for the ith strain
  if rt_array:
    initial_sub_samples = [_n_subdivide(seq, n, False) for seq in full_seqs] # not all elems will have arrays of the same length

    if debug:
      print(f"Initial sub samples:\n {initial_sub_samples}\n")

    # padding section
    max_length = max(len(x) for x in initial_sub_samples) # find the strain with the largest number of subdivisions
    padded = [] # POTENTIAL ISSUE: need to normalize the length, but unsure if model hallucinates embeddings for empty padding inputs
    num_pads = []
    pad_start_indices = []
    for samp in initial_sub_samples:
        assert isinstance(samp, list), f"Samples from initial sub samples is {type(samp)} but should be type list."
        discrepancy = max_length - len(samp)
        padded.append(np.array(samp + [''] * discrepancy)) # add empty padding entries to samp list representing even subdivisions of base pairs
        num_pads.append(discrepancy)
        pad_start_indices.append(len(samp))

    if debug:
      print(f"Num pads:\n {num_pads}\n")

    pads_per_val = dict(zip(d.keys(), num_pads))
    pad_starts = dict(zip(d.keys(), pad_start_indices))
    out = np.array(padded, dtype=object).reshape((len(d), -1))
    # assert len(out.shape) == 2, f"Output has {len(out.shape)} dimension but should have 2 dimensions."
    # assert out.shape[0] == len(dict), f"Output has {out.shape[0]} rows but should {len(dict)}."
    return (out, pads_per_val, pad_starts)

  else:
    sub_samples = list(map(lambda seq: _n_subdivide(seq, n, False), full_seqs))
    pads_per_val = {k: 0 for k in d.keys()}
    pad_starts = {k: len(sub_samples[i]) for i, k in enumerate(d.keys())}
    return (dict(list(zip(d.keys(), sub_samples))), pads_per_val, pad_starts)

def extract_embeddings_prokbert(
    arr: list[list],
    n: int,
    tokenizer: callable,
    model: callable,
    hugface_out_path: str = './experiments',
    hugface_log_path: str = "./experiment_logs", 
    test_mode=False, 
    test_count=3
    ):

  """
  This function first tokenizes a dataset then then extract the embedding representations.
  Takes in an array of dimensions B x d columns: B observations with d subdivision per observation and each element being a string of size n. 
  Outputs a B x d x E tensor for B observations, d subdivisions and a embedding vector encoding semantic value of E per subdivision per observation.
  E is determined by whatever embedding model is being used.
  """
  arr = np.array(arr) # incasd we're given a nested list

  def extract(index, embed_arr):
    """Mutatively appends embed_arr with the the result of passing the ith chunk (arr[:, i]) through the embedding model."""
    nonlocal max_embedding_dim
    i = index
    curr = arr[:, i]
    assert all([isinstance(seq, str) for seq in curr]), f"Not all elements in inputted array are type str."

    # THIS PART IS SPECIFIC TO ProkBERT
    ds = Dataset.from_dict({"base_pairs": curr})

    def tokenize_func(examples, max_length=n):
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
    
    num_gpus = torch.cuda.device_count()
    num_proc = max(1, num_gpus)  # Fallback to 1 if no GPU
    if test_mode:
       print(f"extract_embeddings_prokbert tokenizer map using {num_proc} cores")

    tokenized = ds.map(tokenize_func, batched=True, num_proc=num_proc)

    training_args = TrainingArguments(
    output_dir=hugface_out_path,  # Output directory
    per_device_eval_batch_size=16,  # Batch size for evaluation
    remove_unused_columns=True,  # Ensure compatibility with input format
    logging_dir=hugface_log_path,  # Logging directory
    report_to="none",  # No reporting needed
    )

    # Set up the Trainer for prediction and evaluation
    trainer = Trainer(
        model=model,  # Dummy model
        args=training_args,  # Evaluation arguments
    )
    Y_hat = trainer.predict(tokenized)
    last_hidden_states = Y_hat.predictions[0]

    if last_hidden_states is None:
      raise RuntimeError(f"[ERROR] No output from model on chunk {index}")
    # test
    
    representations = last_hidden_states.mean(axis=1) #NOTE: we perform mean pooling across tokens
    max_embedding_dim = max(max_embedding_dim, representations.shape[1])
    embed_arr.append(representations)

    print(f"{i+1}/{arr.shape[1]} embeddings extracted.")

  embeddings = []
  max_embedding_dim = 0

  # Setup Cuda
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  print(f"Extract_Embeddings Cuda Check:\nModel is on device: {next(model.parameters()).device}")

  print("==============", "BEGINNING EMBEDDING EXTRACTION", "==============")

  try:
    if test_mode:
        print("[DEBUG] Running in TEST MODE")
        for i in range(min(test_count, arr.shape[1])):
            extract(index=i, embed_arr=embeddings)
    else:
        for i in range(arr.shape[1]):
            extract(index=i, embed_arr=embeddings)

    out = np.array(embeddings)
    out = out.transpose(1, 0, 2)

    if out.shape[0] != arr.shape[0] or out.shape[1] != (test_count if test_mode else arr.shape[1]):
        print(f"[WARN] Output shape mismatch: expected {(arr.shape[0], test_count if test_mode else arr.shape[1])}, got {out.shape[:2]}")
  except Exception as e:
      import traceback
      print("[FATAL ERROR] Embedding extraction failed:")
      traceback.print_exc()
      return None
  
  print("==============", "END OF EMBEDDING EXTRACTION", "==============")
  return out

def extract_embeddings_megadna(
    arr: list[list[str]],
    n: int,
    tokenizer: callable,
    model: callable,
    test_mode=False, 
    test_count=3
):
    """
    Given a B x d list of DNA strings of length n, returns B x d x E array of embeddings
    """
    arr = np.array(arr)  # ensure it's a NumPy array
    B, d = arr.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {next(model.parameters()).device}")
    print("============== BEGINNING EMBEDDING EXTRACTION ==============")

    vocab = ['**', 'A', 'T', 'C', 'G', '#']
    nucleotide2token = dict(zip(vocab, range(len(vocab))))
    def encode(seq): return [nucleotide2token[c] for c in seq]

    embeddings = []

    try:
        col_range = range(min(test_count, d)) if test_mode else range(d)

        for i in col_range:
            curr = arr[:, i]
            assert all(isinstance(seq, str) for seq in curr), f"[ERROR] arr[:, {i}] must all be str"

            tokenized = [encode(seq) for seq in curr]  # B x n
            input_tensor = torch.tensor(tokenized).long().to(device)  # shape: B x n

            with torch.no_grad():
                reps = model(input_tensor, return_value='embedding')  # assume returns B x E

            embeddings.append(reps.cpu().numpy())  # d x B x E

            print(f"[INFO] Column {i+1}/{d} done: extracted {reps.shape[-1]}-dim embeddings.")

        out = np.stack(embeddings, axis=1)  # B x d x E

        if out.shape[:2] != (B, len(col_range)):
            print(f"[WARN] Unexpected output shape: {out.shape}")

        print("============== END OF EMBEDDING EXTRACTION ==============")
        return out

    except Exception as e:
        import traceback
        print("[FATAL ERROR] Embedding extraction failed:")
        traceback.print_exc()
        return None

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

def extract_embeddings_glm2(
    arr: list[list[str]],
    n: int,
    tokenizer: callable,
    model: callable,
    test_mode=False, 
    test_count=3,
    device=None,
):
    """
    Extract embeddings from a nested list of genomic sequences using gLM2 model/tokenizer.

    Handles genomic strand tokens <+> and <-> and enforces uppercase for CDS (protein) 
    and lowercase for IGS (nucleotide) sequences.

    Parameters:
    - arr: list of lists of strings (B x d nested list)
    - n: max token length for padding/truncation
    - tokenizer: Huggingface tokenizer callable
    - model: Huggingface model callable
    - test_mode: if True, process only `test_count` subdivisions per observation
    - test_count: number of subdivisions to process if test_mode=True
    - device: torch device (optional). If None, use CUDA if available.

    Returns:
    - numpy array of shape (B, d, E), where E is embedding size after mean pooling
    """
    arr = np.array(arr)  # ensure numpy array for slicing convenience
    B, d = arr.shape
    if test_mode:
        d = min(d, test_count)

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    print(f"[extract_embeddings_glm2] Model on device: {next(model.parameters()).device}")

    def preprocess_sequence(seq: str) -> str:
        if seq.startswith('<+>') or seq.startswith('<->'):
            strand_token = seq[:3]
            content = seq[3:]
            # Heuristic: if content only contains a,c,g,t (case insensitive) → nucleotide (lowercase)
            # else → protein CDS (uppercase)
            if all(c in 'acgtACGT' for c in content):
                content = content.lower()
            else:
                content = content.upper()
            return strand_token + content
        else:
            # If no strand token, return as-is or apply default rule here
            return seq

    embeddings = []

    def extract(index):
        curr_seqs = arr[:, index]
        assert all(isinstance(seq, str) for seq in curr_seqs), f"Non-string found in input at index {index}"

        # Preprocess all sequences in this subdivision
        processed_seqs = [preprocess_sequence(s) for s in curr_seqs]

        ds = Dataset.from_dict({"sequences": processed_seqs})

        def tokenize_func(examples):
            return tokenizer(
                examples["sequences"],
                padding='max_length',
                truncation=True,
                max_length=n,
                return_tensors="pt",
            )
        
        tokenized = ds.map(tokenize_func, batched=True, batch_size=32, remove_columns=["sequences"])

        training_args = TrainingArguments(
            output_dir="./temp_out",
            per_device_eval_batch_size=16,
            remove_unused_columns=True,
            logging_dir="./temp_logs",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
        )

        Y_hat = trainer.predict(tokenized)
        last_hidden_states = Y_hat.predictions

        if last_hidden_states is None:
            raise RuntimeError(f"No output from model on subdivision {index}")

        # Mean pooling across tokens (axis=1)
        pooled = last_hidden_states.mean(axis=1)
        return pooled

    try:
        for i in range(d):
            emb = extract(i)  # shape (B, E)
            embeddings.append(emb)
            print(f"[extract_embeddings_glm2] Extracted embeddings for subdivision {i+1}/{d}")
    except Exception as e:
        import traceback
        print("[FATAL ERROR] Embedding extraction failed:")
        traceback.print_exc()
        return None

    out = np.stack(embeddings, axis=1)  # shape (B, d, E)
    print(f"[extract_embeddings_glm2] Completed. Output shape: {out.shape}")
    return out

   
def extract_embeddings_evo2(
    arr: list[list],
    n: int,
    tokenizer: callable,
    model: callable,
    out_path: str = './experiments',
    log_path: str = "./experiment_logs", 
    test_mode=False
    ):

  """
  This function first tokenizes a dataset then then extract the embedding representations.
  Takes in an array of dimensions B x d columns: B observations with d subdivision per observation and each element being a string of size n. 
  Outputs a B x d x E tensor for B observations, d subdivisions and a embedding vector encoding semantic value of E per subdivision per observation.
  E is determined by whatever embedding model is being used.
  """
  pass
   

