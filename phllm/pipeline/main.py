from datasets import Dataset
from transformers import TrainingArguments, Trainer

import os

from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.config.directory_paths import get_paths
from phllm.extract.chunkers import complete_n_select, extract_embeddings

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
