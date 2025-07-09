# This file is meant to serve as a resource for all the referenced catalogs used in the production of this manuscript:


## Comparison Catalogs
### ZTF (Lucy Lu et al.): https://iopscience.iop.org/article/10.3847/1538-3881/ac9bee#ajac9beet1, MRT Table 1
### TESS (Claytor et al.): https://iopscience.iop.org/article/10.3847/1538-4357/ad159a#apjad159at5, MRT Table 5
### Kepler (Santos et al.): https://iopscience.iop.org/article/10.3847/1538-4365/ac033f#apjsac033ft1, MRT Table 1
### Kbonus (Claytor and Tayar): https://iopscience.iop.org/article/10.3847/1538-4357/add5f0#apjadd5f0t1, MRT Table 1

### Convert MRT Tables to Astropy Tables using the following line
```
from astropy.table import Table
data = Table.read("(#location_of_mrt.txt_file#)", format="ascii.cds")
```

## Sources of Extra Data
### Gaia DR3 (in general, must cite)
### Kepler and TIC IDs have been collated with https://mastweb.stsci.edu/mcasjobs/ (must cite as well)
### XG Boost Info from https://zenodo.org/records/7599789 (must cite Zenodo and paper: https://iopscience.iop.org/article/10.3847/1538-4365/acd53e)
