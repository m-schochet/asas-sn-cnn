from astropy.timeseries import LombScargle
import butterpy as bp
import matplotlib.pyplot as plt
import numpy as np
import os
from pyasassn.lightcurve import LightCurve
from scipy import interpolate
import torch


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

def read_sim(sim_id, sim_dir='files', reset_time=True):
    """ Reads in a butterpy simulation 

    Args:
        sim_id (int): integer number of the butterpy simulation to load 
        sim_dir (str): location of sim files on your machine, preset to files which holds sim 900184 from Schochet & Planet et al. (2025)
        reset_time (boolean): determines whether the time series should be reset in time to begin at a preset minimum time (tmin). True means reset, False means not
    
    Returns:
        lc (butterpy.LightCurve): A butterpy.LightCurve object (see https://github.com/zclaytor/butterpy/blob/main/butterpy/core.py)
    """
    tmin = 2458485
    #sim_path = os.path.join(sim_dir, f"{sim_id//1000:03.0f}", f"sim{sim_id:06d}.fits")
    sim_path = os.path.join(sim_dir, f"sim{sim_id:06d}.fits")
    lc = bp.read_fits(sim_path).lightcurve
    if reset_time:
        lc.time = lc.time - lc.time[0] + tmin
    return lc

def wavelets(self, flux_list=0, min=False, scale=False, tradeoff=2):
    """ Constructs a 2D wavelet-transform and then scales it appropriately for the uses in Schochet et al. 

    Args:
        self (pyasassn.LightCurve): a LightCurve object with which to perform the transform on 
        flux_list (list): either self.flux, or an injected flux list
        min (boolean): whether the transform should be min-masked instead of NaN-masked and then -> 0. (False is NaN mask).
        scale (boolean): whether the transform should be scaled 10x average power and capped at [0, 255] (False is not scaled 10x). 
        tradeoff (int): the same parameter as Γ from LS_wavelet

    Returns:
        masked_wavelet (np.ndarray): A ndarray object that holds the transformed power spectrum
    """

    data = self.data
    x = data.jd
    if flux_list==0:
        y = data.flux
    else:
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
    if (min==True):
        minimum = np.nanmin(wave)
        masked_wavelet = interpolate.griddata((no_mask_x, no_mask_y), newarr.ravel(), (xx, yy), method='cubic', fill_value=minimum)
    else:
        masked_wavelet = interpolate.griddata((no_mask_x, no_mask_y), newarr.ravel(), (xx, yy), method='cubic')
    printer = masked_wavelet[np.newaxis, :]
    power_tensor = torch.tensor(printer)
    
    pooled = torch.nn.functional.adaptive_avg_pool2d(power_tensor, output_size=(64, 64))
    repooled = torch.squeeze(pooled)
    
    no_nans = torch.nan_to_num(repooled, posinf=0, neginf=0)
    scaled_power = no_nans - no_nans.min()
    scaled_power2 = scaled_power * (255/scaled_power.max())
    if (scale==True):
        scaled_power3 = scaled_power2 * 10
        final_wavelet = torch.where(scaled_power3 > 255, 255, scaled_power3)
    else:
        final_wavelet = scaled_power2.numpy().astype("uint8")
    return final_wavelet

def plotter(wavelet, save=False):
    """ Plotting function to display the 2D transformations
    
    Args:
        wavelet (np.ndarray): output 2D transform from wavelets function
        save (boolean, Optional): whether or not the plotted figure should be saved

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10,6), layout='constrained')
    
    tmin = 2458485
    tmax = 2460311
    
    # Axis ticks and labels
    periods = np.arange(1, 30, 0.2275)
    ff = 1/periods
    
    tixs_xaxis = [2458604.5, 2459214.5,  2459792.5, 2460310.5]
    labels_xaxis = ["5/1/2019", "1/1/2021", "8/1/2022", "1/1/2024"]
    
    tixs_yaxisfreq = [ff[0], ff[4], ff[10], ff[120]]
    labels_freqs = [1.0, 0.52, 0.31, 0.04]
    
    labels_time = [1.0, 1.91, 3.27, 28.3]
    
    ax.imshow(wavelet, aspect='auto', extent=(tmin, tmax, np.min(ff), np.max(ff)), vmin=0, vmax=255, rasterized=True)
    fig.supxlabel("Time [d]", weight='bold')
    ax.set_ylabel(r"Frequency [$\mathbf{\frac{1}{day}}$]", weight='bold')
    ax.set_xticks(ticks = tixs_xaxis, labels=labels_xaxis, weight='bold')
    ax.set_yticks(ticks = tixs_yaxisfreq, labels=labels_freqs, weight='bold')
    
    ax2 = ax.twinx()
    ax2.imshow(wavelet, aspect='auto', extent=(tmin, tmax, np.min(ff), np.max(ff)), vmin=0, vmax=255, rasterized=True)
    ax2.set_yticks(ticks = ax.get_yticks(), labels=labels_time, weight='bold')
    ax2.set_ylabel("Period [day]", weight='bold')
    if(save==True):
        plt.savefig("transformation.pdf")
    plt.show()

def reduce(noisy_lc):
    """ Apply the data cleaning reductions from Section 2 of Schochet & Planet et al. (2025) to a light curve

    Args:
        noisy_lc (pyasassn.LightCurve): a LightCurve object with which to reduce the data

    Returns:
        noisy_lc (pyasassn.LightCurve): the reduced LightCurve object

    """

    tmin = 2458485
    tmax = 2460311
    
    noisy_lc.data = noisy_lc.data[noisy_lc.data["phot_filter"] == "g"]
    noisy_lc.data = noisy_lc.data[noisy_lc.data["quality"] == "G"]
    noisy_lc.data = noisy_lc.data[noisy_lc.data["mag_err"] < 99]
    noisy_lc.data = noisy_lc.data[noisy_lc.data["jd"] >= tmin] 
    noisy_lc.data = noisy_lc.data[noisy_lc.data["jd"] <= tmax]
    
    return noisy_lc

def inject_flux(sim, lightcurve):
    """ Inject a simulation into a template light curve given the simulation and LightCurve objects. 
    Algorithm from Appendix A of Schochet & Planet et al. (2025)

    Args:
        sim (butterpy.LightCurve): a butterpy.LightCurve object that holds the simulated light curve 
        lightcurve (pyasassn.LightCurve): a noise template.LightCurve object to inject simulation flux into

    Returns:
        new_flux (list): injected flux values, of the same length as lightcurve
        
    """
    tmin = 2458485
    tmax = 2460311
    sim_window = (tmin <= sim.time) & (sim.time <= tmax)
    sim_time = sim.time[sim_window]
    sim_flux = sim.flux[sim_window]
    new_flux = ((np.interp(lightcurve.jd, sim_time, sim_flux)) * lightcurve.flux) / np.median(sim_flux)
    
    return new_flux