import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import butterpy as bp
from butterpy.distributions import Uniform, LogUniform


def run(jobid):
    #Begin by making the folder and save location for our sims
    np.random.seed(jobid)
    outfile = sys.stdout
    outpath = os.path.join("(#INSERT NAME OF YOUR SAVE FOLDER HERE#)", f"{jobid:03d}")
    os.makedirs(outpath, exist_ok=True)
    
    # Run the BP sims for up to 12 years (ASAS-SN baseline), even though we will eventually downselect an ~5 year baseline for analysis
    f = bp.Flutter(1000, duration=12*365, period=Uniform(1, 30))
    f.to_csv(f"output/sims{jobid:03d}.csv")

    # Print the simulation information as the top header for the output file (tcyc is our actual rotation period of interest)
    print(f"idx,  act, range, tcyc, nspots, fmax", file=outfile)

    for i, row in f.DataFrame.iterrows():
        s = bp.Surface()

        r = s.emerge_regions(
            ndays=f.duration,
            activity_level=row["activity_level"],
            butterfly=row["butterfly"],
            cycle_period=row["cycle_period"],
            cycle_overlap=row["cycle_overlap"],
            max_lat=row["max_lat"],
            min_lat=row["min_lat"])

        nspots = len(r)
        rng = row["max_lat"] - row["min_lat"]

        # Light curve starts anywhere from year 1 to year 7 and lasts 5 years (5-year baseline for ASAS-SN)
        istart = np.random.randint(365/f.cadence, 7*365/f.cadence)
        tstart = istart * f.cadence
        time = np.arange(tstart, tstart+365*5, f.cadence)

        l = s.evolve_spots(
            time=time,
            inclination=row["inclination"], 
            period=row["period"],
            shear=row["shear"], 
            tau_evol=row["tau_evol"],
            alpha_med=0.005)

        fmin = 1-l.flux.min()
        simno = 1000*jobid + i
        
        # Save the activity of the simulation into an output file
        print(f"{simno:6d}, {row['activity_level']:4.2f}, {rng:5.2f}, {row['cycle_period']:4.1f}, {nspots:6d}, {fmin:.2e}",
             file=outfile)
        
        fname = os.path.join(outpath, f"sim{simno:06d}.fits")
        s.to_fits(fname, overwrite=True)


if __name__ == "__main__":
    jobid = int(sys.argv[1])
    run(jobid)
