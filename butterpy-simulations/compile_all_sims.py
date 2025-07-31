# This .py file is meant to help combine all the information for the butterpy simulations into a single .csv file that can be input into
# the neural network, allowing for easy matching of an individual simulation "tcyc" with the ASAS-SN ID that it was injected into

import os
from glob import glob
import pandas as pd

#path locations
src_path = "output"

#save path for the csv
dest_path = "(#outpath variable from butterpy-simulations/run_sims.py#)"

if __name__ == "__main__":
    files = sorted(glob(os.path.join(src_path, "*csv")))
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, index_col="simulation_number")
        df = df.set_index(i*len(df) + df.index)
        df.index.name = "simulation_number"
        dfs.append(df)

    all_csv = pd.concat(dfs)
  
    # This final .csv file will be called "sim_input.csv
    all_csv.to_csv(os.path.join(dest_path, "sim_input.csv"))
    print(len(all_csv), "simulations") # Output a final line to confirm a total of one million simulations
