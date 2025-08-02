# Code for "The ASAS-SN Rotation Period CNN (ARC) I: 200,000+ Deep Learning Inferred Periods of Stellar Variability from The All-Sky Automated Survey for Supernovae"
#### AKA: asas-sn-cnn

### Primary Author and contact for issues: [Meir Schochet](mailto:schoche4@msu.edu)
This repository hosts all the code used in the development of the convolutional neural network from Schochet & Planet et al. (in prep) "The ASAS-SN Rotation Period CNN (ARC) I: 200,000+ Deep Learning Inferred Stellar Rotation Periods from The All-Sky Automated Survey for Supernovae"


## Repository Guide
Inside of this repository is a set of folders. Each of these folders are labeled with an integer that corresponds to a step in the project's workflow, such that each folder appears like

```
[integer]-[name of workflow step]
```

Additionally, each of these workflow steps hosts pairs of .sh/.py files, with the assumption being that these jobs are run into a Linux scheduler. This also explains why contained in each .sh file is a line of:

```
module load conda
```

because this is the standard method of "turning on conda" on the University of Florida HiPerGator computing cluster (learn more about HiPerGator here: https://docs.rc.ufl.edu/). In folders where there are multiple job file pairs, we have included a markdown file which can be referenced to know the order in which the jobs were run.

Furthermore, in each these .py files, pathnames are often reference via the following convention
```
"(# insert path here #)"
```
Any lines in these files which have this convention must be replaced with real paths on your machine before the jobs will properly run. Any place where a savepath for a file in one job is used as a loadpath for that same file in another job, we have done our best to note which exact paths we are referencing.
