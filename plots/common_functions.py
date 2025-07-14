from astropy.timeseries import LombScargle
import butterpy as bp
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyasassn
from pyasassn.lightcurve import LightCurve
from scipy import interpolate
import torch

def read_sim(simfile, reset_time=True):
    """
    Load full light curve from fits file and return time and flux.
    """
    
    sim_path = simfile
    lc = bp.read_fits(sim_path).lightcurve
    if reset_time:
        lc.time = lc.time - lc.time[0] + 2458484.5
    return lc


#this function is essentially pulled from https://github.com/asas-sn/skypatrol/blob/master/pyasassn/wavelet.py
def LS_wavelet(tt, ff, x, y, e_y, gam=2):
    
    """
    Computes a wavelet power spectrum. This is a *SLOW* implementation,
    hopefully to be replaced by something more efficient eventually.
    
    The units of tt and ff are assumed to be such that t * f is dimensionless.
    tt and x are assumed to have the same units.
    
    :param tt: Array of times at which to evaluate wavelet PS
    :param ff: Array of frequencies at which to evaluate wavelet PS
    :param x: Time axis of input time series
    :param y: Dynamical quantity (e.g. fluxes) of input time series
    :param e_y: Measurement errors of input time series
    :param gam: tradeoff parameter between frequency and time resolution
              (by Fourier uncertainty principle). Larger values give
              better frequency resolution.
    
    :return: A numpy array containing the wavelet power spectrum.
    """
    
    tmin = 2458485
    tmax = 2460311
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


def single_wavelet(self, flux_list, tradeoff=2):
    """
    Constructs a 2D wavelet-transform power spectrum of a single LightCurve Object.
    
    :param self: a LightCurve Object (re: https://github.com/asas-sn/skypatrol/blob/master/pyasassn/lightcurve.py / https://github.com/lightkurve/lightkurve/blob/main/src/lightkurve/lightcurve.py)
    :param flux: Array of flux values for the passing of an injected flux array in making a 2D wavelet transform


    :return: power_int, a 64x64 2D torch.Tensor object which holds the reshaped wavelet transform image, normed to values of [0, 255]    
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
    printer = masked_wavelet[np.newaxis, :]
    power_tensor = torch.tensor(printer)
    
    pooled = torch.nn.functional.adaptive_avg_pool2d(power_tensor, output_size=(64, 64))
    repooled = torch.squeeze(pooled)
    
    no_nans = torch.nan_to_num(repooled, posinf=0, neginf=0)
    scaled_power = no_nans - no_nans.min()
    scaled_power2 = scaled_power * (255/scaled_power.max())
    power_int = scaled_power2.numpy().astype("uint8")
    return power_int

def scaled_wavelet(
        self, flux_list,
        tradeoff=2,
    ):
    """
    Constructs a wavelet-transform power spectrum of a single LightCurve Object and scales it 10x its original power into a 64x64 array where any values greater
    than 255 are set to 255 to normalize the array as [0, 255] np.uint8 values.
    
    :param tt: Array of times at which to evaluate wavelet PS
    :param ff: Array of frequencies at which to evaluate wavelet PS

    :return: power_int, a 64x64 2D torch.Tensor object which holds the reshaped wavelet transform image, normed to values of [0, 255] (and with power scaled to 10x normal)
    """
    
    data = self.data
    x = data.jd
    y = flux_list
    e_y = data.flux_err
    
    tt = np.linspace(np.min(x), np.max(x), 128)
    periods = np.arange(1, 30, 0.2275)
    ff = 1/periods
        
    wavelet = LS_wavelet(tt, ff, x, y, e_y, gam=tradeoff)
    
    wave = wavelet.T
    x_list = np.arange(0, wave.shape[1])
    y_list = np.arange(0, wave.shape[0])
    wavelet2 = np.ma.masked_invalid(wave)
    xx, yy = np.meshgrid(x_list, y_list)
    
    no_mask_x = xx[~wavelet2.mask]
    no_mask_y = yy[~wavelet2.mask]
    newarr = wavelet2[~wavelet2.mask]
    masked_wavelet = interpolate.griddata((no_mask_x, no_mask_y), newarr.ravel(), (xx, yy), method='cubic')
    printer = masked_wavelet[np.newaxis, :]
    power_tensor = torch.tensor(printer)
    
    pooled = torch.nn.functional.adaptive_avg_pool2d(power_tensor, output_size=(64, 64))
    repooled = torch.squeeze(pooled)
    
    no_nans = torch.nan_to_num(repooled, posinf=0, neginf=0)
    scaled_power = no_nans - no_nans.min()
    scaled_power2 = scaled_power *  (255/scaled_power.max())
    scaled_power3= scaled_power2 * 10
    checked = torch.where(scaled_power3 > 255, 255, scaled_power)
    power_int = checked.numpy().astype("uint8")
    return power_int

def reduce(noisy_lc):
    """
    Do baseline data reductions by removing data points in the wrong filter/determined to be poor observations/before or after the times of analysis
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
    """
    Inject a simulation into a template light curve given the simulation and lightkurve objects
    """
    tmin = 2458485
    tmax = 2460311
    sim_window = (tmin <= sim.time) & (sim.time <= tmax)
    sim_time = sim.time[sim_window]
    sim_flux = sim.flux[sim_window]
    new_flux = ((np.interp(lightcurve.jd, sim_time, sim_flux)) * lightcurve.flux) / np.median(sim_flux)
    
    return new_flux

def plotter(wavelet, save):
    fig, ax = plt.subplots(figsize=(10,6))
    
    tmin = 2458485
    tmax = 2460311
    
    # Axis ticks and labels
    periods = np.arange(1, 30, 0.2275)
    ff = 1/periods
    
    tixs_xaxis = [2458604.5, 2459214.5,  2459792.5, 2460310.5]
    labels_xaxis = ["5/1/2019", "1/1/2021", "8/1/2022", "1/1/2024"]
    
    tixs_yaxisfreq = [ff[0], ff[4], ff[10], ff[40], ff[120]]
    labels_freqs = [1.0, 0.52, 0.31, 0.09, 0.04]
    
    tixs_yaxisperi = [periods[0], periods[4], periods[10], periods[40], periods[120]]
    labels_time = [1.0, 1.91, 3.27, 10.1, 28.3]
    
    ax.imshow(wavelet, aspect='auto', extent=(tmin, tmax, np.min(ff), np.max(ff)), rasterized=True)
    ax.set_xlabel("Time", weight='bold')
    ax.set_ylabel("Frequency (1/day)", weight='bold')
    ax.set_xticks(ticks = tixs_xaxis, labels=labels_xaxis, weight='bold')
    ax.set_yticks(ticks = tixs_yaxisfreq, labels=labels_freqs, weight='bold')
    
    ax2 = ax.twinx()
    ax2.imshow(wavelet, aspect='auto', extent=(tmin, tmax, np.min(ff), np.max(ff)), rasterized=True)
    ax2.set_yticks(ticks = ax.get_yticks(), labels=labels_time, weight='bold')
    ax2.set_ylabel("Period (days)", weight='bold')
    if(save==True):
    	fig.tight_layout()
    	plt.savefig("transformation.pdf")