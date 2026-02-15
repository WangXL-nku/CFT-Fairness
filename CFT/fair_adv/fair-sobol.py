import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.stats import wasserstein_distance
import argparse
from tqdm import tqdm

from data_loader import create_dataset, prepare_images_from_paths
from cal_concept_celeba import load_model, get_model_layers


def compute_wasserstein_distance(y0, y1):
    y0 = y0.detach().cpu().numpy().flatten()
    y1 = y1.detach().cpu().numpy().flatten()
    return wasserstein_distance(y0, y1)


def compute_fair_sobol(C, Phi, h, S, Y, n_samples=5000, K=None):
    device = next(h.parameters()).device
    N, K = C.shape
    d = Phi.shape[1]
    Phi = torch.from_numpy(Phi).float().to(device) if isinstance(Phi, np.ndarray) else Phi.float().to(device)
    C = torch.from_numpy(C).float().to(device) if isinstance(C, np.ndarray) else C.float().to(device)
    S = torch.from_numpy(S).float().to(device) if isinstance(S, np.ndarray) else S.float().to(device)
    Y = torch.from_numpy(Y).float().to(device) if isinstance(Y, np.ndarray) else Y.float().to(device)

    unique_s = torch.unique(S)
    mask0 = (S == unique_s[0])
    mask1 = (S == unique_s[1])

    base_div = compute_wasserstein_distance(Y[mask0], Y[mask1])

    FS = np.zeros(K)

    M = torch.rand(n_samples, K, device=device)

    M_A = M.clone()
    M_B = M.clone()
    M_B[:, :] = M[:, :]

    Y_tilde = []
    for i in range(n_samples):
        mask_i = M[i:i+1, :]
        A_tilde = (C * mask_i) @ Phi
        with torch.no_grad():
            y_tilde = h(A_tilde).squeeze()
        Y_tilde.append(y_tilde)
    Y_tilde = torch.stack(Y_tilde)

    divs = []
    for i in range(n_samples):
        y = Y_tilde[i]
        d_val = compute_wasserstein_distance(y[mask0], y[mask1])
        divs.append(d_val)
    divs = torch.tensor(divs, device=device)
    var_total = divs.var()

    for k in range(K):
        M_A_k = M_A.clone()
        M_B_k = M_B.clone()
        M_B_k[:, k] = 1 - M_A_k[:, k]

        Y_A = []
        Y_B = []
        for i in range(n_samples):
            mask_a = M_A_k[i:i+1, :]
            mask_b = M_B_k[i:i+1, :]
            A_a = (C * mask_a) @ Phi
            A_b = (C * mask_b) @ Phi
            with torch.no_grad():
                y_a = h(A_a).squeeze()
                y_b = h(A_b).squeeze()
            Y_A.append(y_a)
            Y_B.append(y_b)
        Y_A = torch.stack(Y_A)
        Y_B = torch.stack(Y_B)

        div_A = []
        div_B = []
        for i in range(n_samples):
            da = compute_wasserstein_distance(Y_A[i][mask0], Y_A[i][mask1])
            db = compute_wasserstein_distance(Y_B[i][mask0], Y_B[i][mask1])
            div_A.append(da)
            div_B.append(db)
        div_A = torch.tensor(div_A, device=device)
        div_B = torch.tensor(div_B, device=device)

        FS[k] = (0.5 / n_samples) * torch.sum((div_A - div_B)**2) / var_total

    return FS