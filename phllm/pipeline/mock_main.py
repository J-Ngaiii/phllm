from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import numpy as np
import os
import json
from datetime import datetime

def mock_get_paths(bacteria):
    return "./dummy/strains", "./dummy/phages", "./dummy/strain_embeddings", "./dummy/phage_embeddings"

def mock_rt_dicts(path, strn_or_phg='strain', pad_key=False, seq_report=False):
    if pad_key:
        return [["ATGC" * 1000]], [0]
    return {"id1": "ATGC" * 1000}

def select(data, _n_selector, shape):
    return ["ATGC" * 1000 for _ in range(5)]

def complete_n_select(data, context_window):
    return [["ATGC" * 1000] * 5], [0]

def extract_embeddings(data, context_window, tokenize_func, model):
    return np.random.rand(len(data), 768)

def save_to_dir(dir_path, embeddings, pads, name='ecoli', strn_or_phage='strain'):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(dir_path, exist_ok=True)
    embedding_name = f"{'estrain' if strn_or_phage == 'strain' else 'ephage'}_embed"
    pad_name = f"{'estrain' if strn_or_phage == 'strain' else 'ephage'}_pads"
    np.save(f'{dir_path}/{embedding_name}_{timestamp}.npy', embeddings)
    with open(f'{dir_path}/{pad_name}_{timestamp}.json', 'w') as f:
        json.dump(pads, f)

def main():
    # Mocked config
    LLM = 'distilbert-base-uncased'
    CONTEXT_WINDOW = 4000
    BACTERIA = 'ecoli'
    ECOLI_STRAINS, ECOLI_PHAGES, ECOLI_STRAIN_EMBEDDINGS, ECOLI_PHAGE_EMBEDDINGS = mock_get_paths(BACTERIA)

    ecoli_strains = mock_rt_dicts(ECOLI_STRAINS)
    ecoli_phages = mock_rt_dicts(ECOLI_PHAGES, strn_or_phg='phage')

    tokenizer = AutoTokenizer.from_pretrained(LLM)
    model = AutoModel.from_pretrained(LLM)

    def tokenize_func(examples, max_length=CONTEXT_WINDOW):
        return tokenizer(
            examples["base_pairs"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    flattened_ecoli_strains = select(ecoli_strains, None, (0, 4000))
    flattened_ecoli_phages = select(ecoli_phages, None, (0, 4000))

    ds_strains = Dataset.from_dict({"base_pairs": flattened_ecoli_strains})
    ds_phages = Dataset.from_dict({"base_pairs": flattened_ecoli_phages})

    tokenized_estrains = ds_strains.map(tokenize_func, batched=True, num_proc=1)
    tokenized_ephages = ds_phages.map(tokenize_func, batched=True, num_proc=1)

    estrain_embed = extract_embeddings(flattened_ecoli_strains, CONTEXT_WINDOW, tokenize_func, model)
    ephage_embed = extract_embeddings(flattened_ecoli_phages, CONTEXT_WINDOW, tokenize_func, model)

    estrain_pads = [0] * len(flattened_ecoli_strains)
    ephage_pads = [0] * len(flattened_ecoli_phages)

    save_to_dir(ECOLI_STRAIN_EMBEDDINGS, embeddings=estrain_embed, pads=estrain_pads, name=BACTERIA, strn_or_phage='strain')
    save_to_dir(ECOLI_PHAGE_EMBEDDINGS, embeddings=ephage_embed, pads=ephage_pads, name=BACTERIA, strn_or_phage='phage')

if __name__ == "__main__":
    main()
