#!/usr/bin/env python3
"""
Test SLURM workflow (uses logger)
"""

import os
import argparse
import subprocess
import time
import datetime
import logging

# Compute the absolute path to the shared logs/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# Create a time-stamped log file
log_filename = f"workflow_{time.strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_dir, log_filename)

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path)
        #, logging.StreamHandler() No StreamHandler here to silence terminal logging
    ]
)

logger = logging.getLogger(__name__)

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
            logger.info(f"✅ Stage {stage} appears complete (found: {marker})")
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
        logger.error(f"Error submitting {script_path}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None

def track_stage_duration(stage_num, script_path, conf, dependency=None):
    stage_name = conf.get_stage_names()[stage_num]
    marker_path = conf.get_completion_markers()[stage_num]

    logger.info(f"Submitting Stage {stage_num}: {stage_name}")
    start = time.time()
    job_id = submit_job(script_path, dependency=dependency)
    if not job_id:
        logger.error(f"Stage {stage_num} submission failed")
        return None, 0

    logger.info(f"Stage {stage_num} submitted with Job ID: {job_id}")
    logger.info(f"Waiting for completion marker: {marker_path}")

    while not os.path.exists(marker_path):
        time.sleep(10)  # Poll every 10 seconds

    duration = time.time() - start
    logger.info(f"✅ Stage {stage_num} completed in {str(datetime.timedelta(seconds=round(duration)))}")
    return job_id, duration

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

echo "=== Initializing Environment ==="
module load ml/pytorch
echo "Successfully loaded cluster pytorhc enviornment"
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
module load ml/pytorch
echo "Successfully loaded cluster pytorhc enviornment"
cd {args.root_dir}
pip install -e .
echo "Successfully installed local package"

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
    durations = {}

    logger.info(f"Creating SLURM Scripts in {run_dir}")
    script1 = create_input_test(args, run_dir)
    script2 = create_gpu_test(args, run_dir)

    if args.dry_run:
        logger.info("Dry run: not submitting jobs")
        logger.info(f"Stage 1 script: {script1}")
        logger.info(f"Stage 2 script: {script2}")
        return

    os.chdir(run_dir)

    # Stage 1
    if not conf.check_stage_completion(1):
        job1, dur1 = track_stage_duration(1, "test1.sh", conf)
        job_ids[1] = job1
        durations[1] = dur1
    else:
        logger.info("Stage 1 already complete.")

    # Stage 2
    if not conf.check_stage_completion(2):
        dependency = job_ids.get(1)
        job2, dur2 = track_stage_duration(2, "test2.sh", conf, dependency=dependency)
        job_ids[2] = job2
        durations[2] = dur2
    else:
        logger.info("Stage 2 already complete.")

    logger.info("=== Summary ===")
    logger.info(f"Run directory: {os.path.abspath(run_dir)}")
    logger.info("Monitor with: squeue -u $USER")
    logger.info(f"Check logs in: {os.path.join(run_dir, 'logs')}")

    for stage in durations:
        readable = str(datetime.timedelta(seconds=round(durations[stage])))
        logger.info(f"Stage {stage} duration: {readable}")

if __name__ == "__main__":
    main()
