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
import json

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
    # Convert relevant args to dict
    args_dict = {
        "llm": args.llm,
        "context_window": args.context_window,
        "input_strain": args.input_strain,
        "input_phage": args.input_phage,
        "output_strain": args.output_strain,
        "output_phage": args.output_phage,
        "name_bact": args.name_bact,
        # Add more if needed
    }
    json_args = json.dumps(args_dict)
    
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

echo "=== Initializing Environment ==="
module load ml/pytorch
echo "Successfully loaded cluster pytorhc enviornment"
cd {args.root_dir}
pip install -e .
echo "Successfully installed local package"

python3 -c \"
# running the actual code from a seperate file helps with debugging
import json
from types import SimpleNamespace
from phllm.pipeline import main_slurm

args_dict = json.loads(\\"{json_args}\\")
args = SimpleNamespace(**args_dict)
main_slurm(args)
\"

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
    # parser.add_argument('--environment', default='phage_modeling', help='Conda environment name (default: env_1).')
    parser.add_argument('--root_dir', default='/global/home/users/jonathanngai/main/phllm', help='Root directory of local package with necessary scripts for extracting embeddings.')
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
    print(f"Account: {args.account}")
    print(f"Local package root directory: {args.root_dir}")
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