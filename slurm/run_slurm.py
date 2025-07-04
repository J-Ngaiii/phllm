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
    input_strain = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/strains"
    input_phage = "/global/home/users/jonathanngai/main/phllm/data/raw/ecoli/phages"
    output_strain = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/strains"
    output_phage = "/global/home/users/jonathanngai/main/phllm/data/embeddings/ecoli/phages"
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
    gpu = "gpu:H100:1"

    
    # =============================================
    # WORKFLOW PARAMETERS (with defaults)
    # =============================================
    
    # Script name
    script_name = "embed.py"

    # Configs
    llm = "prokbert"
    context_window = "4000"
    name_bact = "ecoli"
    test_mode = "True"
    
    # Debug options
    dry_run = False                     # Create scripts but don't submit jobs
    
    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", script_name,
        
        # Required arguments
        "--input_strain", input_strain,
        "--input_phage", input_phage,
        "--output_strain", output_strain,
        "--output_phage", output_phage,
        "--output", output_dir,

        # Configs
        "--llm", llm,
        "--context_window", context_window,
        "--name_bact", name_bact,
        "--test_mode", test_mode, 
        
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
    print("SLURM Workflow Submission")
    print("=" * 60)
    print(f"SLURM account:     {account}")
    print(f"partition:         {partition}")
    print(f"qos:               {qos}")
    print(f"gpu:               {gpu}")
    print()

    print("Directory Paths")
    print(f"Input strain dir:  {input_strain}")
    print(f"Input phage dir:   {input_phage}")
    print(f"Output strain dir: {output_strain}")
    print(f"Output phage dir:  {output_phage}")
    print(f"Output directory:  {output_dir}")
    print()
    
    if dry_run:
        print("🧪 DRY RUN MODE - Scripts will be created but not submitted")
        print()
    
    print("Submitting workflow with command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        
        if dry_run:
            print("\n✅ Dry run completed successfully!")
            print("Scripts created in slurm_run_* directory")
        else:
            print("\n✅ Workflow submitted successfully!")
            print("\nMonitor progress with:")
            print("  squeue -u $USER")
            print("  tail -f slurm_run_*/logs/stage*_*.out")
            print("\nExpected runtime: 8-12 hours total (6 sequential stages)")
            print("Estimated cost: ~$200-300")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error submitting workflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())