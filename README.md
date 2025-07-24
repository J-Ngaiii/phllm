# Phage-Host Interaction Large Language Model Embedding Extraction
This repo contains code to flow genome sequences into genomic language models (gLMs) and extract their embeddings into numpy arrays. There is automatic handling to split inputted genome sequences into lengths compliant with context windows.


# Setup (Cloning this repo)
- Run `git clone --recurse-submodules https://github.com/J-Ngaiii/phllm.git`
- Then to instantiate the phllm local package move to the root of the repo and run `pip install .`

# Setup (Enviornment - No Evo2)
- Create conda environment in python 3.11 ()
- Run `pip install -r requirements.txt` from the root of the repository

- If error while installing requirements try
    - installing core packages first: `conda install numpy pandas scikit-learn matplotlib seaborn pyarrow -c conda-forge` then running `pip install -r requirements.txt`
    - installing pyarrow in parricular via conda might be helpful if you're running this on your local machine because Apple Silicon (M1/M2/M3 Macs) runs into issues trying to build pyarrow via pip
    - ensure than numpy has a version older than 2.0 (ie numpy<2.0), this many conflict with spacy which uses thinc and blis, modules that require numpy >=2.0

- If trying to run GPUs consider also running `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` so your enviornment has a CUDA. 
    - note that the cuda version doesn't have to be 11.8 it should be whatever matches with your cluster
    - Lawrenceium's `module avail ml/pytorch` comes with two modules at version 11.7 (`ml/pytorch/2.0.1-py3.11.7` and `ml/pytorch/2.3.1-py3.11.7 (D)`)

# Setup (Enviornment - Yes Evo2)
## Evo2 Requirements
**Prerequisites**
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) >= 2.0.0
- [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main) for optimized attention operations (strongly recommended)

**System requirements**
- [OS] Linux (official) or WSL2 (limited support)
- [GPU] Requires Compute Capability 8.9+ (Ada/Hopper/Blackwell) due to FP8 being required
- [Software]
	- CUDA: 12.1+ (12.8+ for Blackwell) with compatible NVIDIA drivers
	- cuDNN: 9.3+
	- Compiler: GCC 9+ or Clang 10+ with C++17 support
	- Python 3.12 required
  
Check respective githubs for more details about [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main) and how to install them.
We recommend using conda to easily install Transformer Engine. Here is an example of how to install the prerequisites:
- Step 1
```bash 
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers[torch]
```
- Step 2:
```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
pip install flash-attn==2.8.0.post2 --no-build-isolation
```
- While on a GPU node:
```bash
pip3 install --no-build-isolation transformer_engine[pytorch]
pip install evo2
pip install .
```
- Then run: `conda install numpy pandas scikit-learn matplotlib seaborn pyarrow -c conda-forge`
- Before finally running `pip install -r requirements.txt` and `pip install .` from the root of the directory

# How test mode works
- Constrains rt_dicts to only return 3 strains/phages
- Constrains extract_embeddings to only look at the first 3 divisions for all strains/phages in a batch (which will just be 3 strains/phages if rt_dicts is put into test mode)

# Build History
- phllm-0.1.0: first working version that included initial workloop for flowing .fna files into ProkBERT and extracting the embeddings
- phllm-0.1.2: version with working test mode for ProkBERT and initial architecture for Evo2