from datasets import Dataset
from transformers import TrainingArguments, Trainer

import os

from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.config.directory_paths import get_paths
from phllm.extract.chunkers import complete_n_select, extract_embeddings

def workflow(llm, context, strain_in, strain_out, phage_in, phage_out, bacteria = 'ecoli', early_exit = False):  
    # Pulling genomes into dictionaries to load into model
    print("Extracting raw data into dictionaries for processing...")
    ecoli_strains = rt_dicts(path=strain_in, seq_report=True)
    ecoli_phages = rt_dicts(path=phage_in, strn_or_phg='phage', seq_report=True)
    
    if early_exit:
        print("Initiating early exit")
        return
    
    # Setting up model
    print("Setting up model...")
    tokenizer = get_model(llm=llm, rv='tokenizer')
    model = get_model(llm=llm, rv='model')

    def tokenize_func(examples, max_length=context):
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

    # Chunking and Extracting Embeddings
    print("Dividing data into chunks...")
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, context)
    ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, context)

    print("Running embedding model...")
    estrain_embed = extract_embeddings(estrain_n_select, context, tokenize_func, model)
    print(f"Strain embeddings for {bacteria} extracted, dimensions: {estrain_embed.shape}")
    ephage_embed = extract_embeddings(ephage_n_select, context, tokenize_func, model)
    print(f"Strain embeddings for {bacteria} extracted, dimensions: {ephage_embed.shape}")

    # Saving Embeddings to Directory
    save_to_dir(strain_out, embeddings=estrain_embed, pads=estrain_pads, name=bacteria, strn_or_phage='strain')
    save_to_dir(phage_out, embeddings=ephage_embed, pads=ephage_pads, name=bacteria, strn_or_phage='phage')
