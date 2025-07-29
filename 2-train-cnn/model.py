from numpy import pi
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    '''
    Build the Rotation predictor 
    '''
    def __init__(self, c=[8, 16, 32], k=3):
        if isinstance(k, int):
            k = [k]*len(c)
        n_nodes = (64 - (sum(k) - len(k))) * c[-1]

        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(  1,  c[0], k[0], 1) # 62 x 20
        self.conv2 = nn.Conv2d(c[0], c[1], k[1], 1) # 60 x 6 
        self.conv3 = nn.Conv2d(c[1], c[2], k[2], 1) # 58 x 1
        self.fc1 = nn.Linear(n_nodes, 256) #  58 x 32 = 1856
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (1, 3))
        x = self.dropout2(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (1, 3))
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (1, x.shape[-1]))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        output = F.softplus(self.fc3(x))
        return output

def Laplacian_NLL(y_pred, y_true, k=1, reduction='mean'):
    """
    Compute Negative Log Likelihood for Laplacian Output Layer.

    Args:
        y_true: Nxk matrix of (data) target values.
        y_pred: Nx2k matrix of parameters. Each row parametrizes
                k Gaussian distributions with (mean, std).
    """
    means = y_pred[:, :k]
    sigmas = y_pred[:, k:]
    b = sigmas / 1.41421356237 # convert from sigma to b
    term1 = torch.log(2 * b)
    term2 = torch.abs(means - y_true) / b
    nll = term1 + term2
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
