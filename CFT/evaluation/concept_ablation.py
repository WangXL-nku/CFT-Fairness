import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.fairness_metrics import compute_deo


def concept_ablation(C, Phi, h, S, y_true, order, device='cuda'):
    N, K = C.shape
    h = h.to(device)
    C = torch.from_numpy(C).float().to(device)
    Phi = torch.from_numpy(Phi).float().to(device)
    S = torch.from_numpy(S).float().to(device)
    y_true = torch.from_numpy(y_true).float().to(device)

    deo_vals = []
    mask = torch.ones(K, device=device)

    for i, idx in enumerate(order):
        mask[idx] = 0.0
        C_masked = C * mask.unsqueeze(0)
        A_masked = C_masked @ Phi
        with torch.no_grad():
            logits = h(A_masked)
            preds = (logits > 0).float() if logits.size(1) == 1 else logits.argmax(dim=1).float()
        deo = compute_deo(y_true.cpu().numpy(), preds.cpu().numpy(), S.cpu().numpy())
        deo_vals.append(deo)

    return np.array(deo_vals)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=str, required=True)
    parser.add_argument("--Phi", type=str, required=True)
    parser.add_argument("--S", type=str, required=True)
    parser.add_argument("--y", type=str, required=True)
    parser.add_argument("--model_h", type=str, required=True)
    parser.add_argument("--fs_scores", type=str, required=True)
    parser.add_argument("--sobol_scores", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    C = np.load(args.C)
    Phi = np.load(args.Phi)
    S = np.load(args.S)
    y = np.load(args.y)

    model_h = torch.load(args.model_h, map_location=args.device)
    model_h.eval()

    fs_scores = np.load(args.fs_scores)
    sobol_scores = np.load(args.sobol_scores)

    order_fs = np.argsort(fs_scores)[::-1]
    order_sobol = np.argsort(sobol_scores)[::-1]

    deo_fs = concept_ablation(C, Phi, model_h, S, y, order_fs, device=args.device)
    deo_sobol = concept_ablation(C, Phi, model_h, S, y, order_sobol, device=args.device)

    if args.output:
        np.savez(args.output, deo_fs=deo_fs, deo_sobol=deo_sobol)