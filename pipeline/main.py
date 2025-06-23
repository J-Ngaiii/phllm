from datasets import Dataset
from transformers import TrainingArguments, Trainer

import os

from phllm.utils.helpers import rt_dicts, by_row_embedding_saver, save_to_dir
from phllm.config.model_factory import get_model
from phllm.config.directory_paths import get_paths
from phllm.preprocess.selection import select, _n_selector
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
