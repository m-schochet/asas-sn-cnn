from astropy.timeseries import LombScargle
import butterpy as bp
from glob import glob
import numpy as np
import pandas as pd
import polars as pl
import pyasassn
from pyasassn.lightcurve import LightCurve
from scipy import interpolate
import torch
import sys
import os

sim_dir = "(#insert your path for the simulated butterpy light curves here, which is the outpath variable from butterpy-simulations/run_sims.py#)"
saved_wavelets = "(#insert your save path for the injected light curves here, which is from the create_folders.py job#)/training_wavelets"

noisy_csv = "(#insert the location of the 'ids_sim_matched.csv' file output from 0.5-assign-sets/assign_sets.py#)"
clump_df = pd.read_csv(noisy_csv).rename(columns={'noise_id':'asas_sn_id'})
id_list = clump_df.asas_sn_id.to_list()
clump_ids = pd.Series(id_list)

#tmin/tmax values, these one correspond to ~1/1/2019 - ~1/1/2024
tmin = 2458485
tmax = 2460311

#this function is essentially pulled from https://github.com/asas-sn/skypatrol/blob/master/pyasassn/wavelet.py
def LS_wavelet(tt, ff, x, y, e_y, gam=2):
    """ Computes a wavelet power spectrum. This is a *SLOW* implementation, hopefully to be replaced by something more efficient eventually.

    Args:
        tt (np.array): Array of times at which to evaluate wavelet PS
        ff (np.array): Array of frequencies at which to evaluate wavelet PS
        x (np.array): Time axis of input time series
        y (np.array): Dynamical quantity (e.g. fluxes) of input time series
        e_y (np.array): Measurement errors of input time series
        Γ (int): [preset to 2] tradeoff parameter between frequency and time resolution (by Fourier uncertainty principle). Larger values give better frequency resolution.

    Returns:
        acc (np.array):  A numpy array containing the 2D wavelet power spectrum.
    """

    acc = np.full((len(tt), len(ff)), np.nan)
  
    def window(x):
        return np.exp(-x**2/2)

    for j, nu in enumerate(ff):
        dt = gam * (1/nu)
        for i, t in enumerate(tt):
            w = window((x-t)/dt)
            m = np.isfinite(np.nan_to_num(e_y/w, nan=np.inf))
            ls = LombScargle(x[m], y[m], dy=(e_y/w)[m])
            p = float(ls.power(nu, normalization='psd'))
            p /= (np.sqrt(2 * np.pi) * dt)
            acc[i, j] = p
    return acc

def read_sim(sim_id, reset_time=True):
    """ Reads in a butterpy simulation 

    Args:
        sim_id (int): integer number of the butterpy simulation to load 
        reset_time (boolean): determines whether the time series should be reset in time to begin at a preset minimum time (tmin). True means reset, False means not
    
    Returns:
        lc (butterpy.LightCurve): A butterpy.LightCurve object (see https://github.com/zclaytor/butterpy/blob/main/butterpy/core.py)
    """

    sim_path = os.path.join(sim_dir, f"{sim_id//1000:03.0f}", f"sim{sim_id:06d}.fits")
    lc = bp.read_fits(sim_path).lightcurve
    if reset_time:
        lc.time = lc.time - lc.time[0] + tmin
    return lc

def single_wavelet(self, flux_list, tradeoff=2):
    """ Constructs a 2D wavelet-transform and scales it appropriately for the uses in Schochet et al. 

    Args:
        self (pyasasssn.LightCurve): a LightCurve object with which to perform the transform on 
        flux_list (list): either the self.flux list of flux values, or an injected flux list
        tradeoff (int): the same parameter as Γ from LS_wavelet

    Returns:
        masked_wavelet (np.ndarray): A ndarray object that holds the transformed power spectrum
    """

    data = self.data
    x = data.jd
    y = flux_list
    e_y = data.flux_err
    
    # generate the time array of evaluation
    tt = np.linspace(np.min(x), np.max(x), 128)
    
    # the ff array we feed into the LS_wavelet function starts from freq = 1 (one rotation per day) to freq = 1/30 (one rotation every 30 days)
    periods = np.arange(1, 30, 0.2275)
    ff = 1/periods
    
    # generate the wavelet transform    
    wavelet = LS_wavelet(tt, ff, x, y, e_y, gam=tradeoff)
    
    # do 2D interpolative masking to cover with a mask at NaN/-inf/+inf values
    wave = wavelet.T
    x_list = np.arange(0, wave.shape[1])
    y_list = np.arange(0, wave.shape[0])
    wavelet2 = np.ma.masked_invalid(wave)
    xx, yy = np.meshgrid(x_list, y_list)
    no_mask_x = xx[~wavelet2.mask]
    no_mask_y = yy[~wavelet2.mask]
    newarr = wavelet2[~wavelet2.mask]
    masked_wavelet = interpolate.griddata((no_mask_x, no_mask_y), newarr.ravel(), (xx, yy), method='cubic')
    
    return masked_wavelet


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
                        wavelet = single_wavelet(noise_lc, new_flux)
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