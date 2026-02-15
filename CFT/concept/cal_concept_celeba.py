import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import sys

current_file = __file__
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from Craft.craft.craft_torch import Craft, torch_to_numpy
from data_loader import create_dataset, prepare_images_from_paths
import torchvision.models as models


def load_model(model_type, model_path, device, num_classes=None):
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=False)
        if num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=False)
        if num_classes:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval().to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="CRAFT Analysis for Model Interpretation")
    parser.add_argument("-dataset", type=str, default='celeba', choices=['celeba', 'utkface', 'lfw', 'cifar10s'])
    parser.add_argument("-model_type", type=str, default='resnet18', choices=['resnet18', 'vgg16'])
    parser.add_argument("-model_path", type=str, default='model/checkpoint.pth')
    parser.add_argument("-data_path", type=str, default='data/')
    parser.add_argument("-output_dir", type=str, default='results/')

    parser.add_argument("-layer", type=int, default=4, help="Layer to extract features from")
    parser.add_argument("-nmf_dim", type=int, default=40)
    parser.add_argument("-target_class", type=int, default=0)
    parser.add_argument("-image_num", type=int, default=3000)
    parser.add_argument("-nb_crops", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-gpu", type=int, default=0)

    return parser.parse_args()


def get_model_layers(model, model_type):
    if model_type == 'resnet18':
        layers = list(model.children())
        g_layers = layers[:args.layer] + [nn.ReLU()]
        h_layers = layers[args.layer:]
    elif model_type == 'vgg16':
        if args.layer <= len(model.features):
            g_layers = list(model.features[:args.layer]) + [nn.ReLU()]
            h_layers = list(model.features[args.layer:]) + [model.avgpool] + [nn.Flatten()] + list(model.classifier)
        else:
            g_layers = list(model.features) + [model.avgpool, nn.Flatten()] + list(
                model.classifier[:args.layer - len(model.features) - 2])
            h_layers = list(model.classifier[args.layer - len(model.features) - 2:])

    g = nn.Sequential(*g_layers)
    h = nn.Sequential(*h_layers) if h_layers else None

    return g, h


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

    if args.dataset == 'cifar10s':
        num_classes = 2
    elif args.dataset == 'utkface':
        num_classes = 2
    else:
        num_classes = None

    model = load_model(args.model_type, args.model_path, device, num_classes)
    g, h = get_model_layers(model, args.model_type)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    all_images = []
    all_paths = []

    with torch.no_grad():
        for batch_imgs, batch_info in dataloader:
            if isinstance(batch_info, list) or isinstance(batch_info, torch.Tensor):
                if isinstance(batch_info, torch.Tensor):
                    batch_info = [str(i) for i in batch_info.tolist()]
                all_paths.extend(batch_info)
            elif isinstance(batch_info, str):
                all_paths.append(batch_info)

            if isinstance(batch_imgs, list):
                batch_imgs = torch.stack(batch_imgs)

            all_images.append(batch_imgs.to(device))

    images_preprocessed = torch.cat(all_images, dim=0)

    craft = Craft(
        input_to_latent=g,
        latent_to_logit=h,
        number_of_concepts=args.nmf_dim,
        patch_size=64,
        batch_size=args.batch_size,
        device=device
    )

    crops, crops_u, w = craft.fit(images_preprocessed)
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

    importances = craft.estimate_importance(images_preprocessed, class_id=args.target_class)
    images_u = craft.transform(images_preprocessed)

    most_important_concepts = np.argsort(importances)[::-1]

    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{args.model_type}", f"nmf_{args.nmf_dim}")
    os.makedirs(output_dir, exist_ok=True)

    for c_id in most_important_concepts:
        concept_save_path = os.path.join(output_dir, f'Concept_{c_id}.npz')
        id_crops_u = crops_u[:, c_id]
        sorted_indices = np.argsort(id_crops_u)[::-1]
        best_crops_ids = sorted_indices[:args.nb_crops]
        best_crops = crops[best_crops_ids]

        img_indices = [crop_id // 16 for crop_id in best_crops_ids]
        best_imgs = images_preprocessed[img_indices].cpu().numpy()

        if all_paths:
            best_paths = [all_paths[i] for i in img_indices]
        else:
            best_paths = []

        np.savez(
            concept_save_path,
            crops=best_crops,
            best_imgs=best_imgs,
            img_paths=np.array(best_paths),
            importances=id_crops_u,
            concept_importance=importances[c_id]
        )

        print(f"Concept {c_id}, Importance: {importances[c_id]:.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)