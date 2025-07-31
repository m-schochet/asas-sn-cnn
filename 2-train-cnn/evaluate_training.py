import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import ConvNet

run_name = "asassn"
run_number = int(sys.argv[1])
run_path = f"{run_name}_{run_number}"

dataroot = '(#outpath variable from 1-inject-noise/create_folders.py, except the f"{jobid:03d}" part. only up until training_wavelets#)'
datapath = os.path.join(dataroot, "all_wavelets.npy")
outpath = "(#input path to save evaluation output csvs#)"

pmax = int(30)

class WaveletDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path):
        self.data_frame = np.load(data_path, mmap_mode='r')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame[idx].astype('float32') 
        X = torch.tensor(X/255)
        X = torch.unsqueeze(X, 0)
        return X


if __name__ == "__main__":
    data = WaveletDataset(data_path=datapath)
    loader = torch.utils.data.DataLoader(data, batch_size=22)
    ids = np.loadtxt(os.path.join(('#insert save path for this file#'), "0-get-quiescent-bkg/asas-sn_training_stars.txt"), dtype=str)

    if run_number == 0:
        c = [8, 16, 32]
    elif run_number == 1:
        c = [16, 32, 64]
    elif run_number == 2:
        c = [32, 64, 128]
    elif run_number == 3:
        c = [64, 128, 256]

    modelpath = os.path.join(run_path, "models", run_name+".pt")

    model = ConvNet(c=c)

    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for d in loader:
            d = d.to(device, dtype=torch.float)
            output = model(d)
            if cuda:
                output = output.cpu().numpy()
            else:
                output = output.numpy()
            preds.extend(output)

    preds = pmax * np.squeeze(preds)
    output = pd.DataFrame(preds, columns=["period", "sigma"])
    output.index = ids
    output.index.name = "target_id"
    output.to_csv(os.path.join(outpath, f"{run_path}_predictions.csv"))
