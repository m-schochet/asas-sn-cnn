This folder hosts files for downloading milliquas light curves onto your machine. However, as discussed in Schochet & Planet et al. (in prep.)——training the final neural network on template milliquas light curves results in a significantly worse performing network than one that is trained on template star light curves. The ideal replacement selection is stars from ASAS-SN with minimal rotation in their light curves which can then be injected with butterpy simulations.

Red clump and giant stars have been empirically shown to be slowly rotating (Ceillier (2017) found rapid rotation in ~2% of giant stars and ~15% of clump stars less massive than 1.1 solar masses, ref: https://www.aanda.org/articles/aa/full_html/2017/09/aa29884-16/aa29884-16.html)

Furthermore, Claytor et al. 2025 (submitted) attempted to expand the success of Claytor et al. (2024) on TESS data to Kepler data (which does not contain light curves for galaxies similar to TESS). As a result, they decided to curate a set of training templates using "slowly rotating giants and clump stars," selected through Gaia magnitude cutoffs.

Calculating absolute Gaia magnitudes as

$M_{g}$ = g + 5 $\log$(p) - 10

Where g is the provided Gaia absolute G-band magnitude and p is the provided Gaia parralax in milliarcseconds, we can then down select clump stars with absolute G-magnitudes between 0.5 and -1.5. Due to the enormous amount of ASAS-SN stars within these limits (of order 10 million), we can then randomly sample from these stars to select one million appropriate templates. However, this procedure is much more extensive due to the disperse nature of stars within the ASAS-SN databases, and so we recommend contacting the team in charge of ASAS-SN's servers to access the light curves of the clump stars used in our study.

We have included in this folder a .csv file containing the list of ASAS-SN stars used to train our neural network alongside the simulation number they were injected with and which set (training/testing/validating) they were included in. We have also included a .txt file with just the ASAS-SN IDs of the stars used in training our network.
