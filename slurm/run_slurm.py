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
    input_strain = ""
    input_phage = ""
    output_strain = ""
    output_phage = ""
    output_dir = ""
    
    # =============================================
    # SLURM CONFIGURATION
    # =============================================
    account = "ac_mak"
    partition = "lr7"                    # SLURM partition 
    qos = "lr_normal"                    # SLURM QOS
    environment = "phage_modeling"       # Conda environment name

    
    # =============================================
    # WORKFLOW PARAMETERS (with defaults)
    # =============================================
    
    # Configs
    llm = 'prokbert'
    context_window = 4000
    
    # Debug options
    dry_run = False                     # Create scripts but don't submit jobs
    
    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", "slurm_embed.py",
        
        # Required arguments
        "--input_strain", input_strain,
        "--input_phage", input_phage,
        "--output_strain", output_strain,
        "--output_phage", output_phage,
        "--output", output_dir,
        
        # SLURM configuration  
        "--account", account,
        "--partition", partition,
        "--qos", qos,
        "--environment", environment,
    ]
    
    if dry_run:
        cmd.append("--dry_run")
    
    # =============================================
    # SUBMIT WORKFLOW
    # =============================================
    print("=" * 60)
    print("SLURM Workflow Submission")
    print("=" * 60)
    print(f"Input strain:      {input_strain}")
    print(f"Input phage:       {input_phage}")
    print(f"Output directory:  {output_dir}")
    print(f"SLURM account:     {account}")
    print(f"Environment:       {environment}")
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