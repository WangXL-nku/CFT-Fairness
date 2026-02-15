import os
import glob
import numpy as np


def count_adversarial_samples(directory, pattern="*.npz"):
    files = glob.glob(os.path.join(directory, pattern))
    return len(files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    assert len(args.dirs) == len(args.names)

    counts = []
    for d in args.dirs:
        cnt = count_adversarial_samples(d)
        counts.append(cnt)
        print(f"{os.path.basename(d)}: {cnt}")

    if args.output:
        np.savez(args.output, names=args.names, counts=counts)