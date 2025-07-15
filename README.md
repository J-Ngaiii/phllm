# Phage-Host Interaction Large Language Model Embedding Extraction
This repo contains code to flow genome sequences into genomic language models (gLMs) and extract their embeddings into numpy arrays. There is automatic handling to split inputted genome sequences into lengths compliant with context windows.


# Setup (Cloning this repo)
- Run `git clone --recurse-submodules https://github.com/J-Ngaiii/phllm.git`
- Then to instantiate the phllm local package move to the root of the repo and run `pip install .`

# Setup (Enviornment)
- Create conda environment in python 3.11
- Run `pip install -r requirements.txt` from the root of the repository

- If error while installing requirements try
    - installing core packages first: `conda install numpy pandas scikit-learn matplotlib seaborn pyarrow -c conda-forge` then running `pip install -r requirements.txt`
    - installing pyarrow in parricular via conda might be helpful if you're running this on your local machine because Apple Silicon (M1/M2/M3 Macs) runs into issues trying to build pyarrow via pip
    - ensure than numpy has a version older than 2.0 (ie numpy<2.0), this many conflict with spacy which uses thinc and blis, modules that require numpy >=2.0

- If trying to run GPUs consider also running `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` so your enviornment has a CUDA. 
    - note that the cuda version doesn't have to be 11.8 it should be whatever matches with your cluster
    - Lawrenceium's `module avail ml/pytorch` comes with two modules at version 11.7 (`ml/pytorch/2.0.1-py3.11.7` and `ml/pytorch/2.3.1-py3.11.7 (D)`)

# How test mode works
- Constrains rt_dicts to only return 3 strains/phages
- Constrains extract_embeddings to only look at the first 3 divisions for all strains/phages in a batch (which will just be 3 strains/phages if rt_dicts is put into test mode)

# Build History
- phllm-0.1.0: first working version that included initial workloop for flowing .fna files into ProkBERT and extracting the embeddings
- phllm-0.1.2: version with working test mode for ProkBERT and initial architecture for Evo2