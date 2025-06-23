from setuptools import setup, find_packages

setup(
    name="phllm",  # e.g., prokbert-pipeline
    version="0.1.0",
    description="Pipeline for genomic language model embeddings, processing and utilization in predicting phage-host interactions",
    author="Jonathan Ngai",
    packages=find_packages(exclude=["data*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "pyarrow",  
        "datasets",
        "transformers",
        "matplotlib",
        "seaborn",
        "umap-learn",
        "tqdm",
        "biopython",
        "einops",
        "packaging",
        "pre-commit",
        "rich",
        "ruff",
        "setuptools",
        "wheel"
    ],
    python_requires=">=3.11",
)