import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_deltas(directory, max_samples=None):
    deltas = []
    files = glob.glob(os.path.join(directory, "*.npz"))
    if max_samples:
        files = files[:max_samples]
    for f in files:
        data = np.load(f)
        if 'c_orig' in data and 'c_adv' in data:
            delta = data['c_adv'] - data['c_orig']
            deltas.append(delta)
    return np.array(deltas)


def pairwise_cosine_distances(vectors):
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim = np.abs(vectors @ vectors.T)
    dist = 1 - sim
    return dist[np.triu_indices_from(dist, k=1)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    deltas = load_deltas(args.dir, max_samples=args.max_samples)
    print(f"Loaded {len(deltas)} deltas")

    if len(deltas) < 2:
        print("Not enough samples")
        exit()

    dists = pairwise_cosine_distances(deltas)
    print(f"Mean distance: {np.mean(dists):.4f}, Std: {np.std(dists):.4f}")

    if args.output:
        np.save(args.output, dists)