from datasets import Dataset
from transformers import TrainingArguments, Trainer

import os

from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model, get_embedding_extractor, check_status
from phllm.extract.chunkers import complete_n_select

def workflow(llm, context, strain_in, strain_out, phage_in, phage_out, bacteria = 'ecoli', early_exit = False, test_mode=False, test_count=3):  
    """
    Runs the full workflow for extracting genome embeddings using a specified LLM.

    This function loads genomic sequences for bacterial strains and phages, chunks them into context-sized segments,
    runs an LLM-based embedding model, and saves the resulting embeddings to disk.

    Parameters
    ----------
    - llm (str): Name of the large language model to use (e.g., 'prokbert', 'evo2').
    - context (int): Length of each chunk (in tokens) used when splitting genome sequences for embedding.
    - strain_in (str or Path): File path to the input file or directory containing strain sequences.
    - strain_out (str or Path): Output directory path where strain embeddings and metadata will be saved.
    - phage_in (str or Path): File path to the input file or directory containing phage genome sequences.
    - phage_out (str or Path): Output directory path where phage embeddings and metadata will be saved.
    - bacteria (str, optional): Name of the bacterial species for logging and saving (default is 'ecoli').
    - early_exit (bool, optional): If True, exits the function after loading data and before model inference.
    - test_mode (bool, optional): If True, reduces data processed for fast prototyping or testing.

    Returns
    -------
    None
        Saves embedding results to disk. Prints out progress and status messages during execution.
    """
    # Pulling genomes into dictionaries to load into model

    print("Extracting raw data into dictionaries for processing...")
    print("\n")
    ecoli_strains = rt_dicts(path=strain_in, seq_report=True, test_mode=test_mode, test_count=test_count)
    ecoli_phages = rt_dicts(path=phage_in, strn_or_phg='phage', seq_report=True, test_mode=test_mode, test_count=test_count)
    
    if early_exit:
        print("Initiating early exit")
        return
    
    # Setting up model
    print("Setting up model...")
    print("\n")
    check_status(llm=llm)
    tokenizer = get_model(llm=llm, rv='tokenizer')
    print(f"Loaded tokenizer for '{llm}': {tokenizer}")
    model = get_model(llm=llm, rv='model')
    print(f"Loaded model for '{llm}': {model}")


    # Chunking and Extracting Embeddings
    print("Dividing data into chunks...")
    print("\n")
    estrain_n_select, estrain_pads, estrain_pad_indices = complete_n_select(ecoli_strains, context)
    ephage_n_select, ephage_pads, ephage_pad_indices = complete_n_select(ecoli_phages, context)

    print("Running embedding model...")
    print("\n")

    print(f"Dimensions of chunked strain array: {estrain_n_select.shape}")
    embedding_extractor = get_embedding_extractor(llm=llm)
    print(f"Using extractor: {embedding_extractor.__name__}")
    estrain_embed = embedding_extractor(estrain_n_select, context, tokenizer, model, test_mode=test_mode)
    print(f"Strain embeddings for {bacteria} extracted, dimensions: {estrain_embed.shape}")

    print(f"Dimensions of chunked phage array: {ephage_n_select.shape}")
    ephage_embed = embedding_extractor(ephage_n_select, context, tokenizer, model, test_mode=test_mode)
    print(f"Phage embeddings for {bacteria} extracted, dimensions: {ephage_embed.shape}")

    # Saving Embeddings to Directory
    print(f"Initiating saving of embeddings...")
    print("\n")
    save_to_dir(strain_out, embeddings=estrain_embed, pads=estrain_pads, pad_indices=estrain_pad_indices, name=bacteria, strn_or_phage='strain', debug=test_mode)
    save_to_dir(phage_out, embeddings=ephage_embed, pads=ephage_pads, pad_indices=ephage_pad_indices, name=bacteria, strn_or_phage='phage', debug=test_mode)
    print(f"Main workloop finished, exiting function...")
