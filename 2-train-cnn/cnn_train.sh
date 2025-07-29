#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --output=output/%x-%A-%04a.out (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --error=output/%x-%A-%04a.err (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --mem=10G
#SBATCH --gres=gpu:1 ## request a single gpu for each job
#SBATCH --array=0-3

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python -u cnn_train.py $SLURM_ARRAY_TASK_ID
