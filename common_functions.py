import os
import sys
import warnings
from glob import glob
import numpy as np
import polars as pl
import pandas as pd
from pyasassn.lightcurve import LightCurve
import torch
import pyasassn
from scipy import interpolate
import shutil
from astropy.timeseries import LombScargle
import butterpy as bp

def LS_wavelet(tt, ff, x, y, e_y, gam=2):
    # Instantiate accumulating array
    acc = np.full((len(tt), len(ff)), np.nan)

    # We will choose a "wavelet" that is essentially a Gaussian-modulated sinusoid.
    def window(x):
        return np.exp(-x**2/2)
    for j, f in enumerate(ff):
        dt = gam * (1/f)
        for i, t in enumerate(tt):
            w = window((x-t)/dt)
            m = np.isfinite(np.nan_to_num(e_y/w, nan=np.inf))
            ls = LombScargle(x[m], y[m], dy=(e_y/w)[m])
            p = float(ls.power(f, normalization='psd'))
            p /= (np.sqrt(2 * np.pi) * dt)
            acc[i, j] = p
    return acc

def single_wavelet(
        self,
        tradeoff=2,
    ):
        """
        Constructs a wavelet-transform power spectrum of a single LightCurve Object.

        :param tt: Array of times at which to evaluate wavelet PS
        :param ff: Array of frequencies at which to evaluate wavelet PS
        
        """
        
        data = self.data
        x = data.jd
        y = data.flux
        e_y = data.flux_err

        tt = np.linspace(np.min(x), np.max(x), 128)
        periods = np.arange(1, 30, 0.2275)
        ff = 1/periods
            
        #the ff array we feed into the LS_wavelet function starts from freq = 1 meaning 
        #rotates once per day, to freq = 1/30                 
        #where rotates once every 30 days
        
        #generate the wavelet transform    
        wavelet = LS_wavelet(tt, ff, x, y, e_y, gam=tradeoff)
        
        #do 2d interpolative masking to cover the blanks
        
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

def read_sim(sim_dir, sim_id, reset_time=True):
    """Load full light curve from fits file and return time and flux.
    """
    sim_path = os.path.join(sim_dir, f"{sim_id//1000:03.0f}", f"sim{sim_id:06d}.fits")
    lc = bp.read_fits(sim_path).lightcurve
    if reset_time:
        tmin = 2458485
        lc.time = lc.time - lc.time[0] + tmin
    return lc