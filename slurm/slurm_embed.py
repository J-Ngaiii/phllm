#!/usr/bin/env python3
"""
Modified SLURM workflow submission for full_workflow.py
Breaks the workflow into 5 sequential SLURM jobs with proper dependencies.
Stage 5 now includes both k-mer generation AND modeling.
Includes capability to resume from specific stages.
"""

import os
import sys
import argparse
import subprocess
import time

class pipe_config():
    
    def __init__(self, args):
        self.args = args

        output_dir = args.output
        self.completion_markers = {
                1: f"{output_dir}",
            }
        
        self.stage_names = {
            1: "Extract Embeddings", 
        }
        
    def get_stage(self):
        return self.args
    
    def get_stage_names(self):
        return self.stage_names
    
    def get_completion_markers(self):
        return self.completion_markers
    
    def check_stage_completion(self, stage):
        """Check if a stage has been completed based on expected output files."""
        completion_markers = self.get_completion_markers()
        
        marker = completion_markers.get(stage)
        if marker and os.path.exists(marker):
            print(f"✅ Stage {stage} appears complete (found: {marker})")
            return True
        return False

def submit_job(script_path):
    """Submit a SLURM job and return job ID."""
    try:
        result = subprocess.run(['sbatch', '--parsable', script_path], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {script_path}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_full_pipe(args, run_dir):
    """Stage 1: extracting phage and strain data into a tokenized huggingface dataset"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=full_pipe
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres={args.gpu}
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output={run_dir}/logs/pipe_%j.out
#SBATCH --error={run_dir}/logs/pipe_%j.out

echo "=== Stage 1: Tokenizing Data ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURMD_NODENAME, Started: $(date)"

echo "=== Enviornment Setup ({args.environment}) ==="
module load anaconda3
conda activate {args.environment} 2>&1 || {{
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate {args.environment}
}}

echo "=== GPU info ==="
nvidia-smi || echo "No GPUs found or NVIDIA drivers missing"
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
"

python -c "import torch; print(torch.version.cuda)"

python3 -c "
import sys
import os
from datasets import Dataset
from transformers import TrainingArguments, Trainer

from phllm.utils.helpers import rt_dicts, save_to_dir
from phllm.config.model_factory import get_model
from phllm.config.directory_paths import get_paths
from phllm.extract.chunkers import complete_n_select, extract_embeddings

# Setting Variables
LLM = '{args.llm}'
CONTEXT_WINDOW = {args.context_window}
STRAIN_INPUT = '{args.input_strain}'
PHAGE_INPUT = '{args.input_phage}'
STRAIN_OUTPUT = '{args.output_strain}'
PHAGE_OUTPUT = '{args.output_phage}'
BACTERIA = '{args.name_bact}'

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
"

echo "Stage 1 completed: $(date)"
"""
    
    script_path = os.path.join(run_dir, "stage1_preprocessing.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Submit full_workflow.py as 5 sequential SLURM jobs")
    
    # Paths
    parser.add_argument('--input_strain', required=True, help='Input strain FASTA path.')
    parser.add_argument('--input_phage', help='Input phage FASTA path.')
    parser.add_argument('--output_strain', required=True, help='Output strain npz(compressed numpy array) path.')
    parser.add_argument('--output_phage', help='Output phage npz(compressed numpy array) path.')
    parser.add_argument('--output', help='General output directory path.')

    # Configs
    parser.add_argument('--llm', default='prokbert', help='Llm model to use for generating embeddings.')
    parser.add_argument('--context_window',type=int, default=4000, help='Context window for the Embedding model.')
    parser.add_argument('--name_bact', help='Name of bacteria.')
    
    # SLURM-specific arguments
    parser.add_argument('--account', default='ac_mak', help='SLURM account (default: ac_mak).')
    parser.add_argument('--partition', default='es1', help='SLURM partition (default: es1).')
    parser.add_argument('--qos', default='es_normal', help='SLURM QOS (default: es_normal).')
    parser.add_argument('--environment', default='phage_modeling', help='Conda environment name (default: phage_modeling).')
    parser.add_argument('--gpu', help='GPU to be used for the job.')
    parser.add_argument('--dry_run', action='store_true', help='Create scripts but do not submit jobs.')
    
    args = parser.parse_args()
    
    # Create timestamped run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"slurm_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    print(f"=== SLURM Workflow Submission (1 Stage) ===")
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {args.output}")
    print(f"Output Strain directory: {args.output_strain}")
    print(f"Output Phage directory: {args.output_phage}")
    print(f"Account: {args.account}, Environment: {args.environment}")
    print()
    
    # Check which stages are already complete
    print("Checking for completed stages...")
    conf = pipe_config(args=args)
    stage_keys = conf.get_completion_markers()
    print()
    
    # Create all scripts
    print("Creating SLURM job scripts...")
    scripts = {}
    scripts[1] = create_full_pipe(args, run_dir)
    
    if args.dry_run:
        print("Dry run - scripts created but not submitted")
        print("Scripts:")
        for i, script in scripts.items():
            print(f"  Stage {i}: {script}")
        return
    
    # Change to run directory first
    original_dir = os.getcwd()
    run_dir_abs = os.path.abspath(run_dir)
    os.chdir(run_dir)
    
    # Submit jobs with dependencies, starting from specified stage
    job_ids = {}
    
    print("Submitting jobs...")
    
    # Submit starting stage
    # Change back to original directory
    os.chdir(original_dir)
    
    print(f"\n=== Job Submission Summary ===")
    print(f"Run directory: {run_dir_abs}")
    print("Monitor with: squeue -u $USER")
    print("View logs: tail -f logs/stage*_*.out")
    print("Expected total runtime: 2-3 hours (one full pipe)")

if __name__ == "__main__":
    main()