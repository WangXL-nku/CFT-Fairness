import os
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_fpr_fnr(human_labels, model_labels):
    tn, fp, fn, tp = confusion_matrix(human_labels, model_labels, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    human = np.load(args.human)
    model = np.load(args.model)
    assert human.shape == model.shape

    fpr, fnr = compute_fpr_fnr(human, model)
    print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    if args.output:
        np.savez(args.output, fpr=fpr, fnr=fnr)