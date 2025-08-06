import sys
import os
import warnings
from glob import glob
import numpy as np
import pandas as pd


def map_noise(noise_ids:list, N_sims=1000000, weights=[8, 1, 1]): 
    """ This function helps us map the one million "template" IDs to the simulations used in training the neural network

    Args:
        noise_ids (list): A list of template IDs for which to assign to sets
        N_sims (int): (presrt to 1,000,000) the total number of simulations
        weights (list): 3 term list to denote the split in size between training-testing-validation sets

    Returns:
        noise_match (pd.DataFrame): a compiled list of template IDs/sets/and simulation numbers with which to run injection codes
    """
    weights = np.array(weights)/sum(weights) # normalize the weights array

    np.random.seed(88) # ensure the same seed is used each time this code is run
    np.random.shuffle(noise_ids)

    # count training partition simulations
    N_train = int(N_sims * weights[0])
    N_val = int(N_sims * weights[1])
    N_test = N_sims - N_train - N_val

    # compute rough index of separation
    N_noise = len(noise_ids)
    i_train = int(N_noise * weights[0])
    i_val = i_train + int(N_noise * weights[1])

    # get lists of ids
    train_ids = noise_ids[:i_train]
    val_ids = noise_ids[i_train:i_val]
    test_ids = noise_ids[i_val:]

    # fill training ids to reach N_train, N_val, etc.
    N_samples = N_sims // len(noise_ids) # number of times to use each noise template
    noise_match = pd.DataFrame()
    
    for ids, N, name in zip([train_ids, val_ids, test_ids], [N_train, N_val, N_test], ['train', 'val', 'test']):
        tr_ids = np.asarray([[j]*N_samples for j in ids]).flatten()
        tr_ids = np.concatenate([tr_ids, np.random.choice(ids, size=N-len(tr_ids))])
        np.random.shuffle(tr_ids)
        new_df = pd.DataFrame(tr_ids, columns=["noise_id"])
        new_df["set"] = name
        noise_match = pd.concat([noise_match, new_df], ignore_index=True)
        
    noise_match.index.name = "simulation_number"
    return noise_match
    
def main():
    noise_path = np.loadtxt(os.path.join(('#insert save path for this file#'), "0-get-quiescent-bkg/asas-sn_training_stars.txt").tolist()
    noise = map_noise(id_list)
    save_path = "(#INSERT THE LOCATION WHERE YOU WANT TO SAVE THE CSV OF CROSS MATCHED Simulation-IDs-Set GROUPINGS")
    saved_path = os.path.join(save_path, "ids_sim_matched.csv")
    noise.to_csv(saved_path)

if __name__ == "__main__":
    main()
