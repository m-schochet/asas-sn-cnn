# Code for "200,000+ Deep Learning—inferred Periods of Stellar Variability from The All-Sky Automated Survey for Supernovae"

[![Static Badge](https://img.shields.io/badge/Access_The_Paper-maroon?logo=paperswithcode&link=https%3A%2F%2Fiopscience.iop.org%2Farticle%2F10.3847%2F1538-4365%2Fae1ba7)](https://iopscience.iop.org/article/10.3847/1538-4365/ae1ba7)     [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


(Large Data Files) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15848601.svg)](https://doi.org/10.5281/zenodo.15848601)

(Jupyter Notebooks) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17387593.svg)](https://doi.org/10.5281/zenodo.17387593)

### Primary Author and contact for issues: [Meir Schochet](mailto:meir@msu.edu)
This repository hosts all the code used in the development of the convolutional neural network from Schochet & Planet et al. 2026 (ApJS, 282, 10)


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

## Citing
If you use any of the programs in this repository or the associated catalog of stellar variability periods for your work, please cite the original paper. The recommended BibTeX citation is
```
@ARTICLE{2026ApJS..282...10S,
       author = {{Schochet}, Meir E. and {Planet}, Penelope and {Claytor}, Zachary R. and {Tayar}, Jamie and {Feinstein}, Adina D.},
        title = "{200,000+ Deep Learning─inferred Periods of Stellar Variability from the All-Sky Automated Survey for Supernovae}",
      journal = {\apjs},
     keywords = {Stellar rotation, Light curves, Convolutional neural networks, Time series analysis, Irregular cadence, 1629, 918, 1938, 1916, 1953},
         year = 2026,
        month = jan,
       volume = {282},
       number = {1},
          eid = {10},
          doi = {10.3847/1538-4365/ae1ba7},
       eprint = {2509.14423},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJS..282...10S},
}
```
