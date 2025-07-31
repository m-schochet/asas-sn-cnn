#!/usr/bin/bash

#SBATCH --job-name=(INSERT YOUR JOB NAME HERE)
#SBATCH --qos=(INSERT YOUR RESOURCE ALLOCATION NAME HERE)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=(INSERT YOUR EMAIL FOR JOB NOTIFICATIONS HERE)
#SBATCH --output=output/%x-%A-%04a.out (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --error=output/%x-%A-%04a.err (ENSURE YOU HAVE A FOLDER NAMED `output` IN THE LOCATION YOU RUN THIS JOB)
#SBATCH --time=04:00:00
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task                
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G             # 128 GB of RAM total

module load conda
conda activate (NAME OF ENVIRONMENT HOSTING THE PACKAGES IN REQUIREMENTS.TXT)
python download_milliquas.py $SLURM_ARRAY_TASK_ID

