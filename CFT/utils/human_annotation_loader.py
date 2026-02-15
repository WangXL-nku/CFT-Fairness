import numpy as np
import pandas as pd

def load_human_annotation(csv_path):
    df = pd.read_csv(csv_path)
    concept_ids = df['concept_id'].values
    labels = df['bias_label'].values
    return concept_ids, labels

def load_human_annotation_npz(npz_path):
    data = np.load(npz_path)
    return data['concept_ids'], data['labels']