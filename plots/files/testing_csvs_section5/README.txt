Inside this folder are the data files needed to recreate the scatter plots exploring 
predictive efficiency across network tests as in Section 5 of Schochet et al. 

Folder labels are as follows:

{log number of objects network was trained on (5/6)}_
{subset of training templates (clump/milliquas)}_
{repeated training templates (nor/yesr)}_
{scaling used (nan/min)}_all/*.csv

The folders are labeled according to the data files contained within them, 
and in relation to the columns of Table 3 in Schochet et al. 


The csvs of 'simulation_input_30_log{5/6}.csv' hold the simulation input parameters used 
in generating the butterpy simulations, and you need these files to compare 
predictions to the completely certain ground priors. 

Note: The log5 file should only be used on the folders that begin with a 5, while the log6 simulation 
csv file should be used on folders that begin with 6
