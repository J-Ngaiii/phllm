# Phage-Host Interaction Large Language Model Embedding Extraction
This repo contains code to flow genome sequences into genomic language models (gLMs) and extract their embeddings into numpy arrays. There is automatic handling to split inputted genome sequences into lengths compliant with context windows.

# Setup
- Create conda environment in python 3.11
- Run `pip install -r requirements.txt` from the root of the repository

- If error while installing requirements try
    - installing core packages first: `conda install numpy pandas scikit-learn matplotlib seaborn pyarrow -c conda-forge` then running `pip install -r requirements.txt`
    - installing pyarrow in parricular via conda might be helpful if you're running this on your local machine because Apple Silicon (M1/M2/M3 Macs) runs into issues trying to build pyarrow via pip

- If trying to run GPUs consider also running `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` so your enviornment has a CUDA. 
    - note that the cuda version doesn't have to be 11.8 it should be whatever matches with your cluster
    - Lawrenceium's `module avail ml/pytorch` comes with two modules at version 11.7 (`ml/pytorch/2.0.1-py3.11.7` and `ml/pytorch/2.3.1-py3.11.7 (D)`)

# Repository Structure
- phllm/
    - data/
        - embeddings/
        - raw/
    - extract/
        - extract_embed.py
    - models/
        - ProkBERT.py
    - processing/
        - process.py
        - utils.py
    - pipeline/
        - main.py
- setup.py
- slurm/ 
