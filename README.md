# Stellar Rotation Period Prediction from the All-Sky Automated Survey for Supernovae Light Curves using Deep Learning
#### AKA: asas-sn cnn

This repository hosts all the code used in the development of the convolutional neural network from Schochet & Planet et al. (in prep) "Stellar Rotation Period Prediction from ASAS-SN Light Curves using Deep Learning"


## Repository Guide
Inside of this repository is a set of folders. Each of these folders are labeled with an integer that corresponds to a step in the project's workflow, such that each folder appears like

```
[integer]-[name of workflow step]
```

Additionally, each of these workflow steps hosts pairs of .sh/.py files, with the assumption being that these jobs are run into a Linux scheduler. This also explains why contained in each .sh file is a line of:

```
module load conda
```

because this is the standard method of "turning on conda" on the University of Florida HiPerGator computing cluster (learn more about HiPerGator here: https://docs.rc.ufl.edu/)
