import os
import sys
import numpy as np

def run(jobid):
    # Begin by making the folder and save location for our noisy injected curves. 
    # This bit of code will generate a number of folders equal to the number of jobs ran

    outpath = os.path.join("(#path to save injected light curves#)", "training_wavelets", f"{jobid:03d}")
    os.makedirs(outpath, exist_ok=False)

if __name__ == "__main__":
    jobid = int(sys.argv[1])
    run(jobid)
