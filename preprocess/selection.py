

def select(d, selector, selector_args=None):
  """The ecoli_strain dicts have values as lists with elements of those lists being strings corresponding to sequences (sometimes a strain has multiple sequences).
  This function implements a selector to return a 1d list of strings, selecting a sequence for each strain/phage."""
  if selector_args is not None:
    if not isinstance(selector_args, (list, tuple)):
        selector_args = [selector_args]
    return selector(d, *selector_args)
  else:
    return selector(d)
  
# *------------------
#  Specific Selectors
# *------------------

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

# *------------------
#  Checking Functions
# *------------------

def _selector_output_checker(lst, selector_logic, selector_args=None):
  """Checks that a flattened list works according to some designated logic."""
  assert isinstance(lst, list), f"Input lst is a {type(lst)} not a list object."

  if selector_logic == 'first n':
    return _first_n_check(lst, selector_args)
  if selector_logic == 'n select':
    # selector_args[0] should be start, [1] should be n value and [2] should be the dictionary to check against
    return _first_n_check(lst, selector_args[1]) and _n_select_start_check(lst, selector_args[0], selector_args[2])

