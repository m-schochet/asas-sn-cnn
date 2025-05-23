#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --mem=2GB            # Total memory usage for this job is 2GB of RAM
#SBATCH --time=08:00:00
#SBATCH --output=output/%x-%A-%03a.out
#SBATCH --error=output/%x-%A-%03a.err

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python compile_all_sims.py $SLURM_ARRAY_TASK_ID
