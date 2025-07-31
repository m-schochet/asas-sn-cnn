#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --mem=200MB
#SBATCH --time=08:00:00
#SBATCH --output=output/%x-%A-%03a.out
#SBATCH --error=output/%x-%A-%03a.err
#SBATCH --array=0-999%100 

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python run_sims.py $SLURM_ARRAY_TASK_ID
