import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

with open('README.md') as f:
    readme = f.read()

# def update_submodules():
#     base_dir = os.path.dirname(__file__)
#     # Check if the .git folder exists
#     if os.path.exists(os.path.join(base_dir, '.git')):
#         print("Updating git submodules...")
#         # Run submodule init and update for 'evo2' --> it's setup file initializes the submodules it needs
#         subprocess.run(["git", "submodule", "update", "--init", "--recursive", "evo2"], check=True, cwd=base_dir)
#     else:
#         print("No .git directory found; skipping submodule update.")

# def install_all_submodules(root_dir="."):
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         if "setup.py" in filenames:
#             # Avoid reinstalling the root project
#             if os.path.abspath(dirpath) == os.path.abspath("."):
#                 continue
#             print(f"📦 Installing submodule at: {dirpath}")
#             subprocess.run(["pip", "install", "-e", dirpath], check=True)
#             # Optionally also run its submodules if nested
#             subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=dirpath, check=True)


# class CustomInstall(install):
#     def run(self):
#         update_submodules()
#         install_all_submodules()
#         super().run() # main project installs after installing submodules

# class CustomDevelop(develop):
#     def run(self):
#         update_submodules()
#         install_all_submodules()
#         super().run() # main project installs after installing submodules

setup(
    name="phllm",  # e.g., prokbert-pipeline
    version="0.1.3",
    packages=find_packages(),  # can also fo 'include=["phllm", "phllm.*", "evo2.evo2", "evo2.evo2.*"]'
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
        "tqdm", 
        "evo2"
    ],
    cmdclass={
        # 'install': CustomInstall,
        # 'develop': CustomDevelop
    }, 
    python_requires=">=3.11", 
    description="Pipeline for genomic language model embeddings, processing and utilization in predicting phage-host interactions",
    long_description=readme, 
    long_description_content_type="text/markdown",
    author="Jonathan Ngai",
    url='https://github.com/J-Ngaiii/phllm.git'
)