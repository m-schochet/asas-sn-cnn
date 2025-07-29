import os
from tqdm import tqdm
import numpy as np

root = "(#insert your save path for the injected light curves here, from the create_folders.py job#)/training_wavelets"
n_sims = 1_000_000

arrays = np.zeros((n_sims, 64, 64), dtype=np.uint8)
for i in tqdm(range(n_sims)):
    
    try:
        j = int(i//1000)
        arrays[i] = np.load(os.path.join(root, f"{j:03d}", f"noisy{i:06d}_wt.npy"))
        pass
    except:
        print(os.path.join(root, f"{j:03d}", f"noisy{i:06d}_wt.npy"))
        continue
    
    
np.save(os.path.join(root, "all_wavelets.npy"), arrays)
