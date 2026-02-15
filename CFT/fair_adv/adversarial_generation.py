import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from data_loader import create_dataset, prepare_images_from_paths
from cal_concept_celeba import load_model, get_model_layers
from fair_sobol import compute_fair_sobol


def project_l_inf(x_adv, x_orig, eps):
    delta = x_adv - x_orig
    delta = torch.clamp(delta, -eps, eps)
    return x_orig + delta


def pgd_attack(x, g, h, Phi, c_orig, concept_idx, delta_coef, s, step_size=4/255, eps=16/255, iters=100):
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)

    orig_feat = g(x)
    if len(orig_feat.shape) == 4:
        orig_feat = F.adaptive_avg_pool2d(orig_feat, (1, 1)).view(orig_feat.size(0), -1)
    else:
        orig_feat = orig_feat

    with torch.no_grad():
        pred_orig = h(orig_feat).argmax(dim=1)

    c_target = c_orig.clone().detach()
    if s == 0:
        c_target[:, concept_idx] = c_orig[:, concept_idx] - delta_coef
    else:
        c_target[:, concept_idx] = c_orig[:, concept_idx] + delta_coef

    h_target = c_target @ Phi

    for _ in range(iters):
        x_adv.requires_grad_(True)
        feat = g(x_adv)
        if len(feat.shape) == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

        loss = torch.norm(feat - h_target, p=2, dim=1).mean()

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv - step_size * grad.sign()
        x_adv = project_l_inf(x_adv, x, eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    with torch.no_grad():
        feat_adv = g(x_adv)
        if len(feat_adv.shape) == 4:
            feat_adv = F.adaptive_avg_pool2d(feat_adv, (1, 1)).view(feat_adv.size(0), -1)
        pred_adv = h(feat_adv).argmax(dim=1)

    if pred_adv != pred_orig:
        return x_adv.detach()
    else:
        return None


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset_config = {
        'dataset_name': args.dataset,
        'transform': None
    }
    if args.dataset == 'celeba':
        dataset_config.update({
            'base_path': os.path.join(args.data_path, 'CelebA'),
            'target_attr': 10,
            'target_attr_value': args.target_class,
            'image_num': args.image_num,
            'split': 'test'
        })
    elif args.dataset == 'utkface':
        dataset_config.update({
            'root_dir': os.path.join(args.data_path, 'UTKFace'),
            'target_task': 'gender'
        })
    elif args.dataset == 'lfw':
        dataset_config.update({
            'min_faces_per_person': 70,
            'resize': 0.4
        })
    elif args.dataset == 'cifar10s':
        dataset_config.update({
            'root_dir': os.path.join(args.data_path, 'CIFAR10S'),
            'split': 'test',
            'label2': 2,
            'label3': 3
        })

    dataset = create_dataset(**dataset_config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    if args.dataset == 'cifar10s':
        num_classes = 2
    else:
        num_classes = None
    model = load_model(args.model_type, args.model_path, device, num_classes)
    g, h = get_model_layers(model, args.model_type)
    g.eval()
    h.eval()

    Phi = np.load(os.path.join(args.nmf_dir, 'concepts.npy'))
    C = np.load(os.path.join(args.nmf_dir, 'coefficients.npy'))
    concept_importance = np.load(os.path.join(args.nmf_dir, 'concept_importance.npy'))
    concept_bias_labels = np.load(os.path.join(args.nmf_dir, 'concept_bias_labels.npy'))
    S = np.load(os.path.join(args.nmf_dir, 'sensitive_attr.npy'))
    Y = np.load(os.path.join(args.nmf_dir, 'predictions.npy'))

    FS = compute_fair_sobol(C, Phi, h, S, Y, n_samples=5000, K=args.nmf_dim)
    theta = 0.1
    important_biased = np.where((concept_bias_labels == 1) & (FS > theta))[0]
    B = important_biased.tolist()
    top_k = args.top_k

    delta_coef = args.delta_coef
    step_size = args.step_size / 255
    eps = args.eps / 255
    pgd_iters = args.pgd_iters

    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{args.model_type}", "adversarial")
    os.makedirs(output_dir, exist_ok=True)

    for idx, (img, img_path) in enumerate(tqdm(dataloader)):
        img = img.to(device)
        s_val = S[idx]
        c_sample = torch.from_numpy(C[idx:idx+1, :]).float().to(device)

        with torch.no_grad():
            feat = g(img)
            if len(feat.shape) == 4:
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
            pred = h(feat).argmax(dim=1).item()
            y_val = pred

        scores = np.abs(C[idx, B])
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_concepts = [B[i] for i in top_indices]

        for concept_id in top_concepts:
            c_idx = B.index(concept_id) if concept_id in B else None
            if c_idx is None:
                continue
            x_adv = pgd_attack(
                img, g, h, torch.from_numpy(Phi).float().to(device),
                c_sample, concept_id, delta_coef, s_val,
                step_size=step_size, eps=eps, iters=pgd_iters
            )
            if x_adv is not None:
                save_path = os.path.join(output_dir, f"sample_{idx:06d}_concept_{concept_id}.npz")
                np.savez_compressed(
                    save_path,
                    orig_img=img.cpu().numpy(),
                    adv_img=x_adv.cpu().numpy(),
                    img_path=str(img_path[0]) if isinstance(img_path, list) else str(img_path),
                    sensitive_attr=s_val,
                    concept_id=concept_id,
                    orig_label=y_val
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='celeba')
    parser.add_argument("-model_type", type=str, default='resnet18')
    parser.add_argument("-model_path", type=str, default='model/checkpoint.pth')
    parser.add_argument("-data_path", type=str, default='data/')
    parser.add_argument("-nmf_dir", type=str, default='results/nmf_40')
    parser.add_argument("-output_dir", type=str, default='results/')
    parser.add_argument("-nmf_dim", type=int, default=40)
    parser.add_argument("-target_class", type=int, default=0)
    parser.add_argument("-image_num", type=int, default=3000)
    parser.add_argument("-top_k", type=int, default=3)
    parser.add_argument("-delta_coef", type=float, default=0.5)
    parser.add_argument("-step_size", type=float, default=4)
    parser.add_argument("-eps", type=float, default=16)
    parser.add_argument("-pgd_iters", type=int, default=100)
    parser.add_argument("-gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)