import numpy as np
import pandas as pd
import polars as pl
from pyasassn.lightcurve import LightCurve
import torch
import sys
import os
from common_functions import read_sim, single_wavelet

sim_dir = "(#insert your path for the simulated butterpy light curves here, which is the outpath variable from butterpy-simulations/run_sims.py#)"
saved_wavelets = "(#insert your save path for the injected light curves here, which is from the create_folders.py job#)/training_wavelets"

noisy_csv = "(#insert the location of the 'ids_sim_matched.csv' file output from 0.5-assign-sets/assign_sets.py#)"
clump_df = pd.read_csv(noisy_csv).rename(columns={'noise_id':'asas_sn_id'})
id_list = clump_df.asas_sn_id.to_list()
clump_ids = pd.Series(id_list)

#tmin/tmax values, these one correspond to ~1/1/2019 - ~1/1/2024
tmin = 2458485
tmax = 2460311


def pipeline(noise_ids, jobid):
    """Inject all simulated butterpy light curves into their associated clump/quiescent light curved
    This function is written algorithmically. The steps of our pipeline are as follows:
        1. read light curves
            a. read noise light curve
            b. read simulated light curve
        2. inject_noise (flux from simulation gets 'injected' into the template 'quiescent' flux)
        3. wavelet transform the combined injected flux list and the original noise light curve
        4. bin the wavelet to the appropriate shape and format for the CNN (64x64)
        5. take the final the binned wavelet as hlsp-like object

    
    The for loop inside this pipeline is specific for *our* processing of ASAS-SN Light Curves. 
    The entire ASAS-SN catalog was brought onto the University of Florida supercomputer (HiPetGator) 
    in discretized file pairs of index-data files, containing the data on all observed ASAS-SN objects. 
    To perform our injections of butterpy into the clump light curves, we developed this code to loop through
    each of these file pairs and then manually check for the data on ASAS-SN stars that need to be injected with a specific simulation. 
    Your milegae may vary when using this loop, and it may need to be redesigned for other data sets! We provide the loop anyways however, 
    with the expectation that it should at least provide a starting idea of how one might inject these simulations into 'quiescent templates'.

    """
    total_injected = 0
    skipped_ids = []
    skipped_simulations = []
  
    index_base = '(#enter your path to the location of the saved index files#)'

    for filenum in range(0, 1091, 1):
        # read in the index file and manipulate it to have the object ID column named to 'asas_sn_id', and save it as a pandas dataframe
        index_file =  os.path.join(index_base, f"index_{str(filenum).zfill(4)}.parq") # our index files were named "index_0000.parq" through "index_1090.parq"
        index = pl.read_parquet(index_file)
        index_ids = index['asas_sn_id'].unique().to_pandas()
        index_checks = index.unique().to_pandas()
        del index
        
        data_path = '(#enter your path to the location of the saved data files#)'
        data_file = os.path.join(data_path, f"data_{str(filenum).zfill(4)}.parq") # our index files were named "data_0000.parq" through "data_1090.parq"
        data = pl.read_parquet(data_file)
        
        for count, element in enumerate(index_ids): # this loops over all of the asas_sn_ids in the open index file
            if noise_ids[noise_ids.isin([element])].empty: # this means the object in the loop that we are on does not appear in the one-million selected clump stars
                continue
            else: # i.e., this object ID does appear in the one-million clump stars
              
                # locate the associated simulation with this object ID and create the save path for the injected wavelet transform
                simnum = clump_df[clump_df['asas_sn_id'] == element]['simulation_number'].values[0]
                out_stem = os.path.join(saved_wavelets, f"{simnum//1000:03.0f}", f"noisy{simnum:06d}")
                save_array=out_stem+"_wt.npy"
                
                if(os.path.exists(save_array)): #this checks to make sure that we do not repeatedly inject an object if it already exists on our drive
                    print(f"Object ID {element} with sim {simnum} already exists. Skipped", file=sys.stdout)
                    total_injected += 1
                    continue
                else:
                    # if the object does not already exist as a transform in our drive, then we go and manually create the Light Curve object for pass through our 'single_wavelet' function
        
                    # Light Curve meta information
                    ra = index_checks[index_checks['asas_sn_id']==element].ra_deg.values[0]
                    dec = index_checks[index_checks['asas_sn_id']==element].dec_deg.values[0]
                    noisy_lc_meta = pd.DataFrame(data = [[element, ra, dec]], columns=['asas_sn_id', 'ra_deg', 'dec_deg'])
        
                    # Light Curve data information
                    datum = data.filter(pl.col('asas_sn_id') == element).to_pandas()
                    datum = datum[datum["phot_filter"] == "g"] # g-band observations only
                    datum = datum[datum["mag_err"] < 99] # remove "erroneous flux measurements"
                    datum = datum[datum["quality"] == "G"] # only select "Good" observations
                    datum = datum[datum["jd"] >= tmin] # only data after ~1/1/19
                    datum = datum[datum["jd"] <= tmax] # only data before ~1/1/24
                    noise_lc = LightCurve(datum, noisy_lc_meta)
        
                    # Read the simulation in and prepare it for injection
                    sim_lc = read_sim(simnum)
                    sim_window = (tmin <= sim_lc.time) & (sim_lc.time <= tmax)
                    sim_time = sim_lc.time[sim_window]
                    sim_flux = sim_lc.flux[sim_window]
                
                    # perform flux injection
                    new_flux = ((np.interp(noise_lc.jd.values, sim_time, sim_flux)) * noise_lc.flux.values) / np.median(sim_flux)
                
                    
                    # create wavelet transform
                    try:     
                        wavelet = single_wavelet(noise_lc, flux_list=new_flux)
                    except ValueError: # in the event the transform experiences a SciPy Value Error, skip the object and note that it'll need to be replaced later
                        print(f"Scipy value error on transform for clump star {element} and simulation {simnum}. Have recorded and skipping this object (MUST FIX LATER)", file=sys.stdout)
                        skipped_ids.append(element)
                        skipped_simulations.append(simnum)
                        continue
                    transformed = wavelet[np.newaxis, :]
                  
                    # make into a 64x64 image
                    trans_tensor = torch.tensor(transformed)
                    pooledtrans = torch.nn.functional.adaptive_avg_pool2d(trans_tensor, output_size=(64, 64))
                    repooledtrans = torch.squeeze(pooledtrans)
        
                    # do final masking over NaN values that may remain in the transform after doing 2D scipy.interpolate to mask over erroneous values
                    no_nans = torch.nan_to_num(repooledtrans, posinf=0, neginf=0)
        
                    # scale the transform to make into uint8 (values of [0, 255])
                    scaled_power = no_nans - no_nans.min()
                    scaled_power2 = scaled_power * (255/scaled_power.max())
                    power_int = scaled_power2.numpy().astype("uint8")
                    total_injected += 1
        
                    # save the transform
                    np.save(save_array, power_int)
        
                    # (commented out) print statement to affirm the transform went through
                    # print(str(element) +"  " + str(simnum) + " successful injection", file=sys.stdout)
        del data, index_checks, index_ids
        # note the completion of a loop through this file, and how many total of the one-million have been completed
        print(f"Index file {filenum} has been evaluated and contained ids/sims were injected and transformed (total count is now {total_injected})", file=sys.stdout)
        
        skipped_objs_dict = {'asas_sn_id': skipped_ids, 'sim_num': skipped_simulations}
        df2 = pd.DataFrame(skipped_objs_dict)
        df2.to_csv(f"(#insert path where to save the csv file of the skipped over objects which errored#)/skipped_objs_{jobid}.csv") # our output csv were labeled "*/skipped_objs_0" to "*/skipped_objs_43"

def main(nworkers, task_N):
    # this main job loop runs through a subset of 23,000 objects at a time
    # allowing us to parralelize into 44 jobs (44 * 23,000 = 1,012,000), and making the final job (jobid 43) only consist of 11,000 objects to transform
    noise_ids = clump_ids[0+(nworkers*task_N):23000+(nworkers*task_N)] 
    if len(noise_ids) == 0:
        print(f"task {task_N}: empty list.", file=sys.stdout)
        return
    pipeline(noise_ids, task_N)

if __name__ == "__main__":
    nworkers = 23000
    task_N = int(sys.argv[1])
    main(nworkers, task_N)