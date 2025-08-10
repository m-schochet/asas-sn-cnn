import os
import sys
from copy import deepcopy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import ConvNet
from model import Laplacian_NLL as loss_function

batch_size = 100
num_epochs = 500
early_stopping_patience = 20
run_name = "asassn"
run_number = int(sys.argv[1])

data_path = '(#outpath variable from 1-inject-noise/create_folders.py, except the f"{jobid:03d}" part. only up until training_wavelets#)'
sim_input = "(#outpath variable from butterpy-simulations/run_sims.py#)"

output_path = "(#insert here the output folder name for the trained cnn files#)" + run_name + f"_{run_number}"
#output cnns will thus be labeled "asassn0" - "asassn3" for the code as is

pmax = int(30) #output activation cares abt this value, this scales all our periods between 0-1

def scale_data(x):
    return x/pmax

def unscale_data(y):
    return y*pmax

class WaveletDataset(Dataset):
    """ Wavelet Dataset class which holds all our 2D wavelets for feeding into the CNN

    Attributes:
        data_path (str): Path on your machine where the all_wavelets.npy file from combine.py is stored
        mode (string): either train, val, or test

    """

    def __init__(self, data_path, mode):
        """ __init__ method for the WaveletDataset class

        Args:
        data_path (str): Path on your machine where the all_wavelets.npy file from combine.py is stored
        mode (string): either train, val, or test

        """
        data = np.load(
            os.path.join(data_path, f"all_wavelets.npy"),
            mmap_mode='r'
        )

        labels = pd.read_csv(os.path.join("(#dest_path variable from butterpy-simulationss/compile_all_sims.py#)",  "sim_input.csv"))["period"].values
        
        if mode == "train":
            data = data[:800000]
            labels = labels[:800000]
        elif mode == "val":
            data = data[800000:900000]
            labels = labels[800000:900000]
        elif mode == "test":
            data = data[900000:]
            labels = labels[900000:]

        self.data_frame = data
        self.labels = scale_data(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame[idx].astype('float32') 
        X = torch.tensor(X/255)
        X = torch.unsqueeze(X, 0)
        label = torch.tensor(self.labels[idx, np.newaxis])
        return X, label

def accuracy(frac_err, threshold=0.1):
    """ Determines the portion of predictions that are accuracte to within a certain threshold

    Args:
        frac_err (np.array): Array of fractonal errors (|Pred-True|/True)
        threshold (float): Fractional threshold of accuracy (currently preset to 0.1, or 10%)

    Returns:
        sum (float):  An value denoting the portion of predictions that were correct to within a frac_err of the threshold
    """
    
    return sum(frac_err <= threshold) / len(frac_err)

def plot(output, target, epoch, model_name="cnn", mode="train"):
    """ Plotter function for creating in-training plots

    Args:
        output (np.array): Array of predictions from the CNN after an epoch of training
        target (np.array): Array of ground truths for the predictions (same size as output)
        epoch (int): The current epoch 
        model_name (str): Name of our model (preset to "cnn")
        mode (str): What mode the CNN is currently in (preset to "train")

    Returns:
        None
    """

    true_period = unscale_data(target)
    pred_period = unscale_data(output[:, 0])
    frac_err = abs(pred_period - true_period)/true_period

    # predicted period vs. true period
    fig, ax = plt.subplots(figsize=(8, 6))
    if mode == "train":
        vmax = 500
    else:
        vmax = 50
    im = ax.hexbin(true_period, pred_period, mincnt=1, linewidths=0.25, vmax=vmax)
 
    ax.plot([0, pmax], [0, pmax], 'k',
            [0, pmax], [0, pmax], 'w--',
            [0, pmax*0.9], [0, pmax], 'r--',
            [0, pmax], [0, pmax*0.9], 'r--');
    ax.set(xlabel='True Period (days)', ylabel='Predicted Period (days)',
           xlim=(0, pmax), ylim=(0, pmax),
           title=f'{100*accuracy(frac_err, 0.1):.0f}% within 10%, {100*accuracy(frac_err, 0.2):.0f}% within 20%'
       )
    fig.colorbar(im, pad=0, label='Number', extend='max')
    fig.tight_layout();
    fig.savefig(f"{output_path}/plots/{model_name}_{mode}_{epoch:03d}.png")
    plt.close(fig)

def train(model, device, train_loader, val_loader, early_stopping_patience=early_stopping_patience, model_name="cnn"):
    """ Function to train our neural network

    Args:
        model (model.ConvNet): model.ConvNet object which is our instantiated CNN
        device (torch.device): torch.device object that denotes whether the CNN is being run on CPUs ("cpu") or GPUs ("cuda")
        train_loader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object that holds the training set data
        val_loader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object that holds the validation set data
        early_stopping_patience (int): integer that determines the number of epochs to stop training after (after a minimum value for validation loss is output with no further decrease in the validation loss)
        model_name (str): Name of our model (preset to "cnn")

    Returns:
        best_weights (deepcopy of model.state_dict()): model.state_dict object that holds the model weights for our best network
        train_p_loss (list): values of training loss over the entire course of training
        val_p_loss (list): values of validation loss over the entire course of training
    """

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Set learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=3)

    # Training loop
    train_p_loss = []
    val_p_loss = []
    min_loss = 100 #does not matter, is changed immediately
    early_stopping_count = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        p_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        train_p_loss.append(p_loss)
        p_loss = test(model, device, train_loader, epoch, 
                      model_name=model_name, mode="train", make_plot=True, verbose=False)
        p_loss = test(model, device, val_loader, epoch, 
                      model_name=model_name, mode="val", make_plot=True)
        val_p_loss.append(p_loss)
        total_loss = p_loss
        scheduler.step(total_loss)    # learning rate scheduler

        if total_loss < min_loss:
            min_loss = total_loss
            early_stopping_count = 0
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
        else:
            early_stopping_count += 1
            print(f'Early Stopping Count: {early_stopping_count}')
            if early_stopping_count == early_stopping_patience:
                print(f"Early Stopping. Best Epoch: {best_epoch} with loss {min_loss:.4f}.")
                with open(f"best_epoch_{run_number}.txt", "w") as f:
                    print(best_epoch, file=f)
                break    

    torch.save(best_weights, f"{output_path}/models/{model_name}.pt")
    return best_weights, train_p_loss, val_p_loss

def train_epoch(model, device, train_loader, optimizer, epoch):
    """ Functon to advance the network training forward an additional epoch

    Args:
        model (model.ConvNet): model.ConvNet object which is our instantiated CNN
        device (torch.device): torch.device object that denotes whether the CNN is being run on CPUs ("cpu") or GPUs ("cuda")
        train_loader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object that holds the training set data
        optimizer (torch.optim.Adam): torch.optim.Adam object that holds the preset parameters for our Adam optimizer
        epoch (int): The current epoch 

    Returns:
        losses (int): the mean losses of our network in this epoch
    """

    model.train() # Set the model to training mode
    period_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = loss_function(output, target)
        loss.backward()                     # Gradient computation        
        optimizer.step()                    # Perform a single optimization step
        period_losses.append(loss.item())

        if (batch_idx*len(data)) % 10000 == 0:
            print('Train Epoch: {} [{:6d}/{} ({:3.0f}%)]\tPeriod Loss: {:9.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), period_losses[-1]))
    return np.mean(period_losses)


def test(model, device, data_loader, epoch=None, model_name=None, mode=None, make_plot=False, verbose=True):
    """ Functon to test our network's recovery after a training epoch

    Args:
        model (model.ConvNet): model.ConvNet object which is our instantiated CNN
        device (torch.device): torch.device object that denotes whether the CNN is being run on CPUs ("cpu") or GPUs ("cuda")
        data_loader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object that holds the set data
        epoch (int, optional): The current epoch 
        model_name (str, optional): Name of our model 
        mode (str, optional): either train, val, or test
        make_plot (boolean, optional): Whether or not a comparison plot of the predictions versus ground truths should be made 
        verbose (boolean, optional): Whether the mean loss should be output or not

    Returns:
        p_loss (float): the loss of our network in this epoch
    """
        
    model.eval()    # Set the model to inference mode
    p_loss = 0
    targets = []
    preds = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in data_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            targets.extend(target.cpu().numpy())
            preds.extend(output.cpu().numpy())
            p_loss += loss_function(output, target, reduction='sum').item()
    if make_plot:
        preds = np.squeeze(preds)
        targ = np.squeeze(targets)
        means = preds[:, 0]
        sigmas = preds[:, 1]
        plot(preds, targ, epoch, model_name=model_name, mode=mode)

        results = {"Predicted Adjusted Period": means,
                   "Predicted Adjusted StD": sigmas,
                   "True Adjusted Period": targ, 
                   "Predicted Period" : unscale_data(means),
                   "Predicted StD": unscale_data(sigmas),
                   "True Period": unscale_data(targ)}
 
        df = pd.DataFrame(results)

    p_loss /= len(data_loader.dataset)
    
    if verbose:
        print(f'Average test loss: {p_loss:.4f}')
    return p_loss

def predict(model, device, test_loader, verbose=True):
    """ Functon to test our network's improvement on the test set

    Args:
        model (model.ConvNet): model.ConvNet object which is our instantiated CNN
        device (torch.device): torch.device object that denotes whether the CNN is being run on CPUs ("cpu") or GPUs ("cuda")
        test_loader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object that holds the *test* set data
        verbose (boolean, optional): Whether the mean test loss should be output or not

    Returns:
        preds (list): list of predicted output values
        labels (list): list of identifiers for all objects with output predictions (same length as preds)
    """
    model.eval()    # Set the model to inference mode
    preds = []
    labels = []
    predicted = 0
    test_p_loss = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            labels.extend(np.array(target))
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            test_p_loss += loss_function(output, target, reduction='sum').item()

            preds.extend(output.cpu().numpy())
            predicted += len(target)
            #print("Progress: {}/{}".format(predicted, len(test_loader.dataset)))

    test_p_loss /= len(test_loader.dataset)
    if verbose:
        print(f'Average test loss: {test_p_loss:.4f}')

    return np.squeeze(preds), np.squeeze(labels)

def main(channels, kernels=3, model_name='cnn', pretrained_weights=None):
    # Make sure gpu is available
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable. Try again.')

    # Create Datasets
    train_dataset = WaveletDataset(data_path=data_path, mode="train")
    valid_dataset = WaveletDataset(data_path=data_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    device = torch.device("cuda")
    model = ConvNet(channels, kernels).to(device)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    # Training loop
    weights, train_p_loss, val_p_loss = train(model, device, train_loader, val_loader,
        early_stopping_patience=early_stopping_patience, model_name=model_name)

    # Evaluate best-fit model
    model.load_state_dict(weights)
    print("\nFinal Performance!")
    print("Training Set:")
    test(model, device, train_loader, make_plot=False)
    print("Validation Set:")
    test(model, device, val_loader, make_plot=False)
 
    # Plot learning curve
    epoch = 1 + np.arange(len(train_p_loss))
    plt.figure()
    plt.plot(epoch, train_p_loss)
    plt.plot(epoch, val_p_loss, alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.savefig(f"{output_path}/plots/{model_name}_performance.png")
    plt.close()

    np.save(f"{output_path}/losses/{model_name}_train_loss.npy", train_p_loss)
    np.save(f"{output_path}/losses/{model_name}_val_loss.npy", val_p_loss)

    # Predict on test set
    print('\nPrediction on Test set')
    test_dataset = WaveletDataset(data_path=data_path, mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    preds, labels = predict(model, device, test_loader, verbose=True)
    means = preds[:, 0]
    sigmas = preds[:, 1]

    results = {"Predicted Adjusted Period": means,
               "Predicted Adjusted StD": sigmas,
               "True Adjusted Period": labels, 
               "Predicted Period" : unscale_data(means),
               "Predicted StD": unscale_data(sigmas),
               "True Period": unscale_data(labels)}
 
    df = pd.DataFrame(results)
    df.to_csv(f"{output_path}/output/{model_name}_predictions.csv", index=False)

if __name__ == '__main__':
    channels = {0: [ 8, 16, 32],
                1: [16, 32, 64],
                2: [32, 64, 128],
                3: [64, 128, 256]}
    for folder in ['models', 'plots', 'losses', 'output']:
        path = os.path.join(output_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    c = channels[run_number]
    print(f'Training with channels {c}')
    main(c, model_name=run_name)
