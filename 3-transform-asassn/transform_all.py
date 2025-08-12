import os
import sys
import numpy as np
import torch
import polars as pl
import pandas as pd
from pyasassn.lightcurve import LightCurve
from common_functions import single_wavelet

def run(jobid):
    jobid = int(jobid)
    index_path = "(#the new_path variable from 3-transform-asassn/index_fix.py#)"
    data_path = "(#the location of the data files=#)"
    save_path = "(#the location to save all of the transformed files#)"

    data_files = os.path.join(data_path, f"data_{str(jobid).zfill(4)}.parq")
    index_files = os.path.join(index_path, f"index_extra_quality_{str(jobid).zfill(4)}.parquet")
    tmin = 2458485
    tmax = 2460311
    
    savepath = os.path.join(save_path, f"{str(jobid).zfill(4)}")
    try:
        os.makedirs(savepath, exist_ok=False)
        print("Folder was made", file=sys.stdout)
    except:
        print("Folder already exists for this job", file = sys.stdout)
    
    index = pl.read_parquet(index_files)
    data = pl.read_parquet(data_files)
    
    unique = data.select("asas_sn_id").unique().sort(by='asas_sn_id')
    unique
    
    for val in range(len(unique)):
        id = unique[val].item()
        strid = str(id)
        
        out_stem = os.path.join(save_path, f"{str(jobid).zfill(4)}", f"{id:06d}")
        save_array=out_stem+"_wt.npy"
        
        index_pandas = index.filter((pl.col('asas_sn_id') == (id))).to_pandas()
        
        catalog = index_pandas[index_pandas.asas_sn_id == id].catalog_sources.iloc[0]
        level = index.filter((pl.col('asas_sn_id') == (id)))[0, 5]

        if(os.path.exists(save_array) == True):
            print(f"File already exists for ASAS-SN ID: {strid} so its skipped")
            continue
        else:
            selected = data.filter((pl.col('asas_sn_id').eq(id)) & (pl.col('quality').eq('G')) & (pl.col('phot_filter').eq("g")) & (pl.col('mag_err') < 99 ) & (pl.col('jd') >= tmin ) & (pl.col('jd') <= tmax))
            data_pandas = selected.to_pandas()
            concat_data = pd.concat([data_pandas])
            ra=float(index_pandas[index_pandas.asas_sn_id == id].ra_deg.iloc[0])
            dec=float(index_pandas[index_pandas.asas_sn_id == id].dec_deg.iloc[0])
            lc_meta = pd.DataFrame(data = [[id, ra, dec]], columns=['asas_sn_id', 'ra_deg', 'dec_deg'])
            noise_lc = LightCurve(concat_data, lc_meta)
            if(len(selected) < 150):       # if there are too few flux points, throw out this object (but note the number of data points and output that)
                index = index.with_columns(pl.when(pl.col("__index_level_0__")==(level)).then(False).otherwise(pl.col('keeper')).alias('keeper'))
                index.write_parquet(index_files)
                print(f"LC of ASAS-SN ID: {strid} had only {str(len(noise_lc.data))} data points, so it has been excluded. This is recorded in the new index file.", file=sys.stdout)
                continue
            else:
                if 'stellar_main' in catalog:
                    print(f"ASAS-SN ID: {strid} is a star, so it should be transformed.", file=sys.stdout)
                    try:
                        wavelet = single_wavelet(noise_lc)
                    except ValueError:
                        print("Scipy Value Error on transform, ASAS-SN ID:  " + strid + " is skipped, and this noted in index.", file=sys.stdout)
                        index = index.with_columns(pl.when(pl.col("__index_level_0__")==(level)).then(False).otherwise(pl.col('keeper')).alias('keeper'))
                        index.write_parquet(index_files)
                        continue
                    except FileNotFoundError:
                        print("Scipy FileNotFoundError on griddata.interpolate, ASAS-SN ID:  " + strid + " is skipped, and this noted in index.", file=sys.stdout)
                        index = index.with_columns(pl.when(pl.col("__index_level_0__")==(level)).then(False).otherwise(pl.col('keeper')).alias('keeper'))
                        index.write_parquet(index_files)
                        continue
                    printer = wavelet[np.newaxis, :]
                    power_tensor = torch.tensor(printer)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(power_tensor, output_size=(64, 64))
                    repooled = torch.squeeze(pooled)
                    no_nans = torch.nan_to_num(repooled, posinf=0, neginf=0)
                    scaled_power = no_nans - no_nans.min()
                    scaled_power2 = scaled_power * (255/scaled_power.max())
                    final_wavelet = scaled_power2.numpy().astype("uint8")
                    np.save(save_array, final_wavelet)
                    print(f"ASAS-SN ID: {strid} was successfully transformed.", file=sys.stdout)
                else:
                    print(f"ASAS-SN ID: {strid} is NOT A STAR, noted in index", file=sys.stdout)
                    index = index.with_columns(pl.when(pl.col("__index_level_0__")==(level)).then(False).otherwise(pl.col('keeper')).alias('keeper'))
                    index.write_parquet(index_files)
                    continue

if __name__ == "__main__":
    jobid = int(sys.argv[1])
    run(jobid)