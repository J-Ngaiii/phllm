from phllm.pipeline.main import workflow
    
if __name__ == "__main__":
    workflow(
        llm='prokbert', 
        context=4000,
        strain_in='/Users/jonathanngai/Desktop/MLy/LBNL-local/phllm/data/raw/ecoli/strains', 
        strain_out='/Users/jonathanngai/Desktop/MLy/LBNL-local/phllm/data/outputs/ecoli/strains', 
        phage_in='/Users/jonathanngai/Desktop/MLy/LBNL-local/phllm/data/raw/ecoli/phages', 
        phage_out='/Users/jonathanngai/Desktop/MLy/LBNL-local/phllm/data/outputs/ecoli/phages', 
        bacteria='ecoli', 
        test_mode=True
    )