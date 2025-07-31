import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from glob import glob
from model import ConvNet

run_name = "asassn"
run_number = int(sys.argv[1])
run_saver = f"{run_name}_{run_number}"

run_path = "(#output_path variable from 2-train-cnn/cnn_train)"
pmax = int(30)

class WaveletDataset(Dataset):
    """
    Wavelet transform dataset <'class'>

    used to compile our transforms and load them into the CNN for evaluation
    
    """

    def __init__(self, data_path):
        self.data_frame = np.sort(np.load(data_path, mmap_mode='r'))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame[idx].astype('float32') 
        X = torch.tensor(X-X.min())
        X = X/X.max()
        X = torch.unsqueeze(X, 0)
        return X

if __name__ == "__main__":
    for i in range(0, 1090, 1):
        # this runs out to 1090 due to the ASAS-SN datafiles used for this work being subdivided into 
        # 1091 index/data file pairs. Loop length can adjust depending on source data size at any future time.
        
        try:
            os.chdir("(#save path for evaluation csv files#)")
            os.mkdir(f"{str(i).zfill(4)}")
            os.chdir(f"{str(i).zfill(4)}")
            outpath = os.curdir
            print("Directory created successfully", file=sys.stdout)
        except FileExistsError:
            print("Directory already exists", file=sys.stdout)
            os.chdir(f"{str(i).zfill(4)}")
            outpath = os.curdir
            
        dataroot = os.path.join("(#save_path variable from 3-transform-asassn/transform_all.py#)", str(i).zfill(4))
        datapath = os.path.join(dataroot, "all_wavelets.npy")

        data = WaveletDataset(data_path=datapath)
        loader = torch.utils.data.DataLoader(data, batch_size=22)

        list_of_files = sorted(glob(os.path.join(dataroot, "*.npy")))
        list_of_filenames = [file for file in list_of_files if 'all_wavelets' not in file]
        ids = []
        for obj in lister:
            #NOTE: This is a janky loop meant to collect the ID for each object in the all_wavelets file, 
            #      by scraping them from the file names themselves (poor programming in retrospect)
            #      as a result, these lines NEED to be adjusted depending on your save paths and the 
            #      total # of characters you are trying to identify before the ID in the filename
            
            index = obj.find("(#text in path that comes right before the file with an object's ID in the name#)/")
            # the string input above will be x characters long + 1 from the forward slash, call this intA
            index2 = obj.find("_wt")
            
            string_id = str(obj[index+('{#number of characters in intA #})' + 5):index2])
            # the +5 here accounts for the added 4 digits of the datapath from the str(i).zfill(4) call and another forward slash to the final folder
            
            ids.append(string_id)
    
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
        output.to_csv(os.path.join(outpath, f"{run_saver}_{i}_predictions.csv"))
