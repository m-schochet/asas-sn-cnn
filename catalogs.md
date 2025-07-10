# This file is meant to serve as a resource for all the referenced catalogs used in the production of this manuscript:


## Comparison Catalogs
### ZTF (Lucy Lu et al. 2922): https://iopscience.iop.org/article/10.3847/1538-3881/ac9bee#ajac9beet1, MRT Table 1
### TESS (Claytor et al. 2024): https://iopscience.iop.org/article/10.3847/1538-4357/ad159a#apjad159at5, MRT Table 5
### Kepler (Santos et al. 2021): https://iopscience.iop.org/article/10.3847/1538-4365/ac033f#apjsac033ft1, MRT Table 1
### Kepler (Santos et al. 2019): https://iopscience.iop.org/article/10.3847/1538-4365/ab3b56#apjsab3b56t3, MRT Table 3
### Kbonus (Claytor and Tayar 2025): https://iopscience.iop.org/article/10.3847/1538-4357/add5f0#apjadd5f0t1, MRT Table 1
### For the two other used TESS catalogs (Holcomb et al. 2022 and Colman et al. 2024), these datasets are not publically available online. As a result, please reach out to those researchers individually if you would like to access these catalogs

### Convert MRT Tables to Astropy Tables using the following line
```
from astropy.table import Table
data = Table.read("(#location_of_mrt.txt_file#)", format="ascii.cds")
data.write('(#save location of the MRT as a.csv#)', format='ascii.csv', overwrite=True)
```

## Sources of Extra Data
### Gaia DR3
### Kepler and TIC IDs have been collated with https://mastweb.stsci.edu/mcasjobs/
### XG Boost information was gathered from https://zenodo.org/records/7599789
