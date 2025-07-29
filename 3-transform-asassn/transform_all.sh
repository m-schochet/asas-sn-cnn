#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --output=output/%x-%A-%04a.out (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --error=output/%x-%A-%04a.err (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --time=4-00:00:00 ## This job is *extremely* computationally expensive due to the current Lomb Scargle implementation being the Press and Rybicki algorithm. Faster speeds may be achieved using an NFFT version)

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python -u transform_all.py $SLURM_ARRAY_TASK_ID


