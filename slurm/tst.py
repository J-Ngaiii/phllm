#!/usr/bin/env python3
"""
Test SLURM workflow (uses logger)
"""

import os
import sys
import time
import datetime
import subprocess
import argparse

# test

class pipe_config():
    def __init__(self, args):
        self.args = args
        output_dir = args.output
        self.completion_markers = {
            1: os.path.join(output_dir, "stage1_complete.txt"),
            2: os.path.join(output_dir, "stage2_complete.txt"),
        }
        self.stage_names = {
            1: "Input Test",
            2: "GPU Test"
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
    """
    Submit a SLURM job and return the job ID string.

    Parameters:
    - script_path (str): Path to the SLURM job script file.
    - dependency (str or None): Optional SLURM job ID to add a dependency on.

    Returns:
    - job_id (str or None): Job ID string if submission succeeded, else None.
    """
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd += ['--dependency', f'afterok:{dependency}']
    cmd.append(script_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        print(f"Submitted job {job_id} with script {script_path}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {script_path}: {e.stderr.strip()}")
        return None


def create_input_test(args, run_dir):
    script = f"""#!/bin/bash
#SBATCH --job-name=stage1_input_test
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={args.gpu}
#SBATCH --time=1:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

echo "=== Test 1: Input Validity ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURMD_NODENAME, Started: $(date)"

echo "=== Initializing Environment ==="
module load ml/pytorch
echo "Successfully loaded cluster pytorch enviornment"
cd {args.root_dir}
pip install -e .
echo "Successfully installed local package"

echo "=== Test Argument Inputs ==="
python3 -c "
test_string = '{args.test_str}'
test_num = {args.test_num}
output_dir = '{args.output}'
print('Test String type check:', type(test_string))
print('Test Num type check:', type(test_num))
print('Output Directory type check:', type(output_dir))
print('Hi', test_string)
print(test_num * 10)
"

touch {args.output}/stage1_complete.txt
"""
    path = os.path.join(run_dir, "test1.sh")
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path

def create_gpu_test(args, run_dir):
    script = f"""#!/bin/bash
#SBATCH --job-name=stage2_gpu_test
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={args.gpu}
#SBATCH --time=1:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

echo "=== Test 2: GPU Test ==="

echo "=== Initializing Environment ==="
module load ml/pytorch/2.3.1-py3.11.7
echo "Successfully loaded cluster pytorhc enviornment"
cd {args.root_dir}
pip install -e .
echo "Successfully installed local package"

echo "=== SLURM Info ==="
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "=== Python Cuda Check ==="
nvidia-smi || echo "No GPUs found"
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
"

python3 -c "
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

for _ in range(100):
    torch.matmul(a, b)
print('GPU stress test complete')
"

touch {args.output}/stage2_complete.txt
"""
    path = os.path.join(run_dir, "test2.sh")
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path

def main():
    parser = argparse.ArgumentParser(description="Submit 2-stage SLURM test workflow")

    parser.add_argument('--test_str', required=True)
    parser.add_argument('--test_num', type=int, required=True)
    parser.add_argument('--output', required=True)

    parser.add_argument('--account', default='ac_mak')
    parser.add_argument('--partition', default='es1')
    parser.add_argument('--qos', default='es_normal')
    parser.add_argument('--root_dir', default='/global/home/users/jonathanngai/main/phllm')
    parser.add_argument('--gpu', default='gpu:1')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"slurm_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    conf = pipe_config(args)
    job_ids = {}

    # Submitting Scripts
    stage1_path = create_input_test(args, run_dir=run_dir)
    stage2_path = create_gpu_test(args, run_dir=run_dir)

    # Stage 1
    if not conf.check_stage_completion(1):
        print("Stage 1 not complete, submitting job...")
        job1_id = submit_job(stage1_path)
    else:
        print("Stage 1 already complete.")

    # Stage 2
    if not conf.check_stage_completion(2):
        print("Stage 2 not complete, submitting job...")
        job2_id = submit_job(stage2_path)
    else:
        print("Stage 2 already complete.")

    print("=== Summary ===")
    print(f"Run directory: {os.path.abspath(run_dir)}")
    print("Monitor jobs with: squeue -u $USER")
    print(f"Check logs in: {os.path.join(run_dir, 'logs')}")

if __name__ == "__main__":
    main()
