import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

with open('README.md') as f:
    readme = f.read()

def update_submodules():
    base_dir = os.path.dirname(__file__)
    # Check if the .git folder exists
    if os.path.exists(os.path.join(base_dir, '.git')):
        print("Updating git submodules...")
        # Run submodule init and update for 'vortex'
        subprocess.check_call(['git', 'submodule', 'init', 'vortex'], cwd=base_dir)
        subprocess.check_call(['git', 'submodule', 'update', 'vortex'], cwd=base_dir)
    else:
        print("No .git directory found; skipping submodule update.")

class CustomBuild(_build_py):
    def run(self):
        update_submodules()
        super().run()

setup(
    name="phllm",  # e.g., prokbert-pipeline
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "umap-learn",
        "torch",
        "datasets",
        "transformers",
        "biopython",
        "tqdm"
    ],
    # cmdclass={
    #     'build': CustomBuild
    # }, 
    python_requires=">=3.11", 
    description="Pipeline for genomic language model embeddings, processing and utilization in predicting phage-host interactions",
    long_description=readme, 
    long_description_content_type="text/markdown",
    author="Jonathan Ngai",
    url='https://github.com/J-Ngaiii/phllm.git'
)