#!/bin/bash
#SBATCH --job-name=cv_capital_common
#SBATCH --time=1-00:00:00        # Maximum allowed time (96 hours / 4 days)
#SBATCH --cpus-per-task=32        # Requesting 32 CPU cores
#SBATCH --mem=20G                # Maximum available memory
#SBATCH --partition=msismall      
#SBATCH --account=funkr           # Using your allocated Slurm account
#SBATCH --mail-user=hassa940@umn.edu  # Your email for job notifications
#SBATCH --mail-type=ALL           # Send email on job BEGIN, END, FAIL

# Load Python module
module load conda

conda activate new_env

# Run your Python script
python cv_capital_common.py