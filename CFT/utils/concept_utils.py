import numpy as np
import os

def load_concept_data(concept_dir, nmf_dim):
    phi_path = os.path.join(concept_dir, 'concepts.npy')
    c_path = os.path.join(concept_dir, 'coefficients.npy')
    imp_path = os.path.join(concept_dir, 'concept_importance.npy')
    pred_path = os.path.join(concept_dir, 'predictions.npy')
    sens_path = os.path.join(concept_dir, 'sensitive_attr.npy')
    bias_path = os.path.join(concept_dir, 'concept_bias_labels.npy')

    Phi = np.load(phi_path) if os.path.exists(phi_path) else None
    C = np.load(c_path) if os.path.exists(c_path) else None
    importance = np.load(imp_path) if os.path.exists(imp_path) else None
    predictions = np.load(pred_path) if os.path.exists(pred_path) else None
    sensitive = np.load(sens_path) if os.path.exists(sens_path) else None
    bias_labels = np.load(bias_path) if os.path.exists(bias_path) else None

    return Phi, C, importance, predictions, sensitive, bias_labels

def compute_relevance_score(C, concept_idx):
    return np.abs(C[:, concept_idx])

def get_top_concepts_for_sample(C, sample_idx, concept_list, k=3):
    scores = np.abs(C[sample_idx, concept_list])
    top_indices = np.argsort(scores)[-k:][::-1]
    return [concept_list[i] for i in top_indices]