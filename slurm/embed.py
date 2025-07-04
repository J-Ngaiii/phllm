#!/usr/bin/env python3
"""
Main Embedding SLURM workflow
"""

import os
import argparse
import subprocess
import time

# test

class pipe_config():
    def __init__(self, args):
        self.args = args
        output_dir = args.output
        self.completion_markers = {
            1: os.path.join(output_dir, "stage1_complete.txt"),
        }
        self.stage_names = {
            1: "Workflow", 
        }

    def get_stage_names(self):
        return self.stage_names

    def get_completion_markers(self):
        return self.completion_markers

    def check_stage_completion(self, stage):
        marker = self.completion_markers.get(stage)
        if marker and os.path.exists(marker):
            print(f"✅ Stage {stage} appears complete (found: {marker})")
            return True
        return False

def submit_job(script_path, dependency=None):
    """Submit a SLURM job and return job ID."""
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd += ['--dependency', f'afterok:{dependency}']
    cmd.append(script_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {script_path}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_workflow(args, run_dir):
    script = f"""#!/bin/bash
#SBATCH --job-name=stage1_input_test
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={args.gpu}
#SBATCH --time=3:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

echo "=== Beginning Main Workflow ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURMD_NODENAME, Started: $(date)"

echo "=== Initializing Environment ==="
module load ml/pytorch
echo "Successfully loaded cluster pytorhc enviornment"
cd {args.root_dir}
pip install -e .
echo "Successfully installed local package"

echo "=== GPU info ==="
nvidia-smi || echo "No GPUs found"
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
"

echo "=== Workflow Begins ==="
python3 -c "
from phllm.pipeline.main import workflow
workflow(
    llm='{args.llm}', 
    context={args.context_window},
    strain_in='{args.input_strain}', 
    strain_out='{args.output_strain}', 
    phage_in='{args.input_phage}', 
    phage_out='{args.output_phage}', 
    bacteria='{args.name_bact}'
)
"

echo "Final output directory contents:"
ls -lh {args.output_strain}
ls -lh {args.output_phage}

echo "=== Workflow Complete ==="

touch {args.output}/workflow_complete.txt

"""
    path = os.path.join(run_dir, "test1.sh")
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path

def main():
    parser = argparse.ArgumentParser(description="Submit embed.py as 1 SLURM job")
    
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
    parser.add_argument('--root_dir', default='/global/home/users/jonathanngai/main/phllm')
    # parser.add_argument('--environment', default='phage_modeling', help='Conda environment name (default: phage_modeling).')
    parser.add_argument('--gpu', default='gpu:1', help='GPU to be used for the job.')
    parser.add_argument('--dry_run', action='store_true', help='Create scripts but do not submit jobs.')
    
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"slurm_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    conf = pipe_config(args)
    job_ids = {}

    print(f"\n=== Creating SLURM Scripts in {run_dir} ===")
    script1 = create_workflow(args, run_dir)

    if args.dry_run:
        print("Dry run: not submitting jobs")
        print("Scripts created:")
        print("  Stage 1:", script1)
        return

    os.chdir(run_dir)

    print("\nSubmitting Stage 1: Input Test")
    if not conf.check_stage_completion(1):
        job1 = submit_job("test1.sh")
        job_ids[1] = job1
        print(f"Stage 1 submitted with Job ID: {job1}")
    else:
        print("Stage 1 already complete.")

    print("\n=== Summary ===")
    print("Run directory:", os.path.abspath(run_dir))
    print("Monitor with: squeue -u $USER")
    print("Check logs in:", os.path.join(run_dir, "logs"))

if __name__ == "__main__":
    main()
