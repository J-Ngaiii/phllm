# Setup
- Create conda environment in python 3.11
- Run `pip install -r requirements.txt` from the root of the repository

- If error while installing requirements try
    - installing core packages first: `conda install numpy pandas scikit-learn matplotlib seaborn pyarrow -c conda-forge` then running `pip install -r requirements.txt`
    - installing pyarrow in parricular via conda might be helpful if you're running this on your local machine because Apple Silicon (M1/M2/M3 Macs) runs into issues trying to build pyarrow via pip

# Repository Structure
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