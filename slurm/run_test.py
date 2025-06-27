#!/usr/bin/env python3
"""
Submit SLURM test workflow script.
Run: python3 submit_slurm_test.py
"""

import subprocess
import sys

def main():
    # =============================================
    # TEST INPUTS - EDIT THESE
    # =============================================
    test_str = "phage-buddy"
    test_num = 42
    output_dir = "/global/home/users/jonathanngai/main/phllm/data/test_outputs"
    
    # =============================================
    # SLURM CONFIGURATION
    # =============================================
    account = "ac_mak"
    partition = "es1"                  # SLURM partition 
    qos = "es_normal"                  # SLURM QOS
    # environment = "env_1"             # Conda environment name
    root_dir = "/global/home/users/jonathanngai/main/phllm"
    gpu = "gpu:H100:1"
    
    # =============================================
    # TEST OPTIONS
    # =============================================
    test_script = "tst.py"   # Name of the SLURM test script
    dry_run = False                          # Set to True to only generate scripts
    
    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", test_script,
        
        "--test_str", test_str,
        "--test_num", str(test_num),
        "--output", output_dir,

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
    # SUBMIT TEST WORKFLOW
    # =============================================
    print("""
    ============================================================
    SLURM Test Workflow Submission
    ============================================================
    Test string:       {test_str}
    Test number:       {test_num}
    Output directory:  {output_dir}
    Project root path: {root_dir}
    SLURM account:     {account}
    Partition:         {partition}
    qos:               {qos}
    GPU:               {gpu}

    Submitting test workflow with command:
    {command}
    """.format(
        test_str=test_str,
        test_num=test_num,
        output_dir=output_dir,
        root_dir=root_dir,
        account=account,
        partition=partition,
        qos=qos, 
        gpu=gpu, 
        command="python3 tst.py " + " ".join(sys.argv[1:])
    ))
    
    if dry_run:
        print("🧪 DRY RUN MODE - Scripts will be created but not submitted")
        print()
    
    print("Submitting test workflow with command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        
        if dry_run:
            print("\n✅ Dry run completed successfully!")
            print("Scripts created in slurm_run_* directory")
        else:
            print("\n✅ Test workflow submitted successfully!")
            print("Monitor with:")
            print("  squeue -u $USER")
            print("  tail -f slurm_run_*/logs/stage*_*.out")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error submitting test workflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
