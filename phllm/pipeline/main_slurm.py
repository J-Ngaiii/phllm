from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.extract.chunkers import complete_n_select, extract_embeddings

# Setting Variables
def main_slurm(args):
    LLM = args.llm
    CONTEXT_WINDOW = args.context_window
    STRAIN_INPUT = args.input_strain
    PHAGE_INPUT = args.input_phage
    STRAIN_OUTPUT = args.output_strain
    PHAGE_OUTPUT = args.output_phage
    BACTERIA = args.name_bact

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