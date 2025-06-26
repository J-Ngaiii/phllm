import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phllm')))
phllm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phllm'))
print('path: ', phllm_path)
from datasets import Dataset
from transformers import TrainingArguments, Trainer

from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.config.directory_paths import get_paths
from phllm.extract.chunkers import complete_n_select, extract_embeddings



# Setting Variables
LLM = "prokbert"
CONTEXT_WINDOW = 4000
STRAIN_INPUT = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/strains"
PHAGE_INPUT = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/phages"
STRAIN_OUTPUT = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/strains"
PHAGE_OUTPUT = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/phages"
BACTERIA = "ecoli"

# Pulling genomes into dictionaries to load into model
ecoli_strains = rt_dicts(path=STRAIN_INPUT, seq_report=True)
ecoli_phages = rt_dicts(path=PHAGE_INPUT, strn_or_phg='phage', seq_report=True)

# Setting up model
tokenizer = get_model(llm=LLM, rv='tokenizer')
model = get_model(llm=LLM, rv='model')

def tokenize_func(examples, max_length=CONTEXT_WINDOW):
    return tokenizer(
        examples['base_pairs'],  # input a list of multiple strings you want to tokenize from a huggingface Dataset object
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Chunking and Extracting Embeddings
estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, CONTEXT_WINDOW)
ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, CONTEXT_WINDOW)

estrain_embed = extract_embeddings(estrain_n_select, CONTEXT_WINDOW, tokenize_func, model)
ephage_embed = extract_embeddings(ephage_n_select, CONTEXT_WINDOW, tokenize_func, model)

# Saving Embeddings to Directory
save_to_dir(STRAIN_OUTPUT, embeddings=estrain_embed, pads=estrain_pads, name=BACTERIA, strn_or_phage='strain')
save_to_dir(PHAGE_OUTPUT, embeddings=ephage_embed, pads=ephage_pads, name=BACTERIA, strn_or_phage='phage')