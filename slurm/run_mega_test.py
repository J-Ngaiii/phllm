#!/usr/bin/env python3
"""
Complete SLURM workflow submission script with all parameters.
Edit the paths and parameters below, then run: python3 submit_my_workflow.py
"""

import subprocess
import sys

def main():
    # =============================================
    # YOUR PATHS - EDIT THESE
    # =============================================
    output_dir = "/global/home/users/jonathanngai/main/phllm/data/outputs/ecoli"
    root_dir = "/global/home/users/jonathanngai/main/phllm"
    
    # =============================================
    # SLURM CONFIGURATION
    # =============================================
    account = "ac_mak"
    partition = "es1"                    # SLURM partition 
    qos = "es_normal"                    # SLURM QOS
    # environment = "env_1"       # Conda environment name
    root_dir = "/global/home/users/jonathanngai/main/phllm"
    gpu = "gpu:H100:2"

    
    # =============================================
    # WORKFLOW PARAMETERS (with defaults)
    # =============================================
    
    # Script name and Environment
    script_name = "megaDNA_debug.py"
    environment = "phllm0.1.4"

    # Configs
    
    # Debug options
    dry_run = False                     # Create scripts but don't submit jobs
    
    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", script_name,
        
        # Required arguments
        "--environment", environment, 
        "--output", output_dir,

        # Configs
        
        # SLURM configuration  
        "--account", account,
        "--partition", partition,
        "--qos", qos,
        "--root_dir", root_dir, 
        # "--environment", environment,
        "--gpu", gpu
    ]
    
    if dry_run:
        cmd.append("--dry_run")
    
    # =============================================
    # SUBMIT WORKFLOW
    # =============================================
    print("=" * 60)
    print("SLURM megaDNA Testflow Submission")
    print("=" * 60)
    print(f"SLURM account:     {account}")
    print(f"partition:         {partition}")
    print(f"qos:               {qos}")
    print(f"gpu:               {gpu}")
    print()

    print("Directory Paths")
    print(f"Environment:  {environment}")
    print()
    
    print("Directory Paths")
    print(f"Output directory:  {output_dir}")
    print()
    
    if dry_run:
        print("🧪 DRY RUN MODE - Scripts will be created but not submitted")
        print()
    
    print("Submitting testflow with command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        
        if dry_run:
            print("\n✅ Dry run completed successfully!")
            print("Scripts created in slurm_run_* directory")
        else:
            print("\n✅ Testflow submitted successfully!")
            print("\nMonitor progress with:")
            print("  squeue -u $USER")
            print("  tail -f slurm_run_*/logs/stage*_*.out")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error submitting workflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())