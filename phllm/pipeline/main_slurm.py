import torch
from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.extract.chunkers import complete_n_select, extract_embeddings

# Setting Variables
def main_slurm(
        llm = 'prokbert', 
        context_window = 4000, 
        input_strain = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/strains", 
        input_phage = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/phages", 
        output_strain = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/strains", 
        output_phage = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/phages", 
        name_bact = "ecoli"
    ):
    # Main Cude Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Main Cuda Check:\nModel is on device: {next(model.parameters()).device}")

    # Pulling genomes into dictionaries to load into model
    ecoli_strains = rt_dicts(path=input_strain, seq_report=True)
    ecoli_phages = rt_dicts(path=input_phage, strn_or_phg='phage', seq_report=True)

    # Setting up model
    tokenizer = get_model(llm=llm, rv='tokenizer')
    model = get_model(llm=llm, rv='model')

    def tokenize_func(examples, max_length=context_window):
        return tokenizer(
            examples['base_pairs'],  # input a list of multiple strings you want to tokenize from a huggingface Dataset object
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    # Chunking and Extracting Embeddings
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, context_window)
    ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, context_window)

    estrain_embed = extract_embeddings(estrain_n_select, context_window, tokenize_func, model)
    ephage_embed = extract_embeddings(ephage_n_select, context_window, tokenize_func, model)

    # Saving Embeddings to Directory
    save_to_dir(output_strain, embeddings=estrain_embed, pads=estrain_pads, name=name_bact, strn_or_phage='strain')
    save_to_dir(output_phage, embeddings=ephage_embed, pads=ephage_pads, name=name_bact, strn_or_phage='phage')