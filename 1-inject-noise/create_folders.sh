#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --mem=2GB
#SBATCH --time=01:00:00
#SBATCH --output=output/%x-%A-%03a.out
#SBATCH --error=output/%x-%A-%03a.err
#SBATCH --array=0-999%100

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python -u create_folders.py $SLURM_ARRAY_TASK_ID
