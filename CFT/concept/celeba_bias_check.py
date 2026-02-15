import os
import numpy as np
import torch
from math import ceil
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from craft.craft_torch import Craft, torch_to_numpy

attributes = ['', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def load_model(model_path, device):
    model = torch.load(model_path)
    model.eval().to(device)
    return model


def prepare_images(image_paths, transform):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
    return torch.stack(images, 0)


def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5063, 0.4258, 0.3832], [0.2669, 0.2456, 0.2413])
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Celeba Image Processing")
    parser.add_argument("-base_save_path", type=str, default='/media/data0/wzy/', help="Base save path for other files")
    # parser.add_argument("-model_path", type=str, default='makeup')
    parser.add_argument("-layer", type=int, default=4, help="Directory to save figures")
    parser.add_argument("-target_attr", type=int, default=19)
    parser.add_argument("-target_attr_value", type=int, default=1)
    parser.add_argument("-image_num", type=int, default=3000)
    # parser.add_argument("-crops_num", type=int, default=20)
    parser.add_argument("-nb_crops", type=int, default=200, help="Number of crops to display")
    parser.add_argument("-batch_size", type=int, default=512 * 4)
    parser.add_argument("-gpu", type=int, default=1)
    print(parser.parse_args())
    return parser.parse_args()

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.base_save_path, 'Craft/models/celeba_models/', attributes[args.target_attr],
                              'celeba_resnet50.pth')
    model = load_model(model_path, device)
    transform = create_transform()

    # Processing
    attr_file_path = os.path.join(args.base_save_path, 'dataset/CelebA/Anno/list_attr_celeba.txt')
    partition_file_path = os.path.join(args.base_save_path, 'dataset/CelebA/Eval/list_eval_partition.txt')
    test_images_dir = os.path.join(args.base_save_path, 'dataset/CelebA/test')

    target_attr_v = 1 if args.target_attr_value == 1 else -1

    with open(partition_file_path, 'r') as file:
        eval_lines = file.readlines()

    with open(attr_file_path, 'r') as file:
        lines = file.readlines()[2:]

    assert len(eval_lines) == len(lines)

    target_image_paths = []
    un_corr_img_paths = []
    for i in range(len(eval_lines)):
        image_set = eval_lines[i].strip().split()[1]
        attr_line_split = lines[i].strip().split()
        img_name = attr_line_split[0]
        target_label = int(attr_line_split[args.target_attr])
        if image_set == '2' and target_label !=target_attr_v and len(un_corr_img_paths) < args.image_num:
            image_path = os.path.join(test_images_dir, img_name)
            un_corr_img_paths.append(image_path)
        if image_set == '2' and target_label == target_attr_v:
            image_path = os.path.join(test_images_dir, img_name)
            target_image_paths.append(image_path)
            if len(target_image_paths) == args.image_num:
                break

    images_preprocessed = prepare_images(target_image_paths, transform)
    uncorr_images_preprocessed = prepare_images(un_corr_img_paths, transform)
    # g = nn.Sequential(*(list(model.children())[:4])) # input to penultimate layer
    # h = lambda x: model.head.fc(torch.mean(x, (2, 3))) # penultimate layer to logits
    # g = nn.Sequential(*(list(model.children())[:4]))  # input to penultimate layer
    # h = lambda x: model.head.fc(torch.mean(x, (2, 3)))  # penultimate layer to logits
    ll = list(model.children())[:args.layer]
    ll.append(nn.ReLU())
    g = nn.Sequential(*(ll))
    # g = nn.Sequential(*(list(model.children())[:layer4]))
    h = nn.Sequential(*(list(model.children())[args.layer:]))

    craft = Craft(input_to_latent=g,
                  latent_to_logit=h,
                  number_of_concepts=20,
                  patch_size=64,
                  batch_size=args.batch_size,
                  device=device)

    crops, crops_u, w = craft.fit(images_preprocessed)
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

    importances = craft.estimate_importance(images_preprocessed, class_id=args.target_attr_value)
    uncorr_u = craft.transform(uncorr_images_preprocessed)
    uncorr_u = np.mean(uncorr_u, axis=(1,2))
    img_u = craft.transform(images_preprocessed)
    img_u = np.mean(img_u, axis=(1,2))
    corr_mean = np.mean(img_u, axis=0)
    corr_std = np.std(img_u, axis=0)
    uncorr_mean = np.mean(uncorr_u, axis=0)
    uncorr_std = np.std(uncorr_u, axis=0)
    for i in range(uncorr_u.shape[1]):
        print("Concept", i, " , Importance:", round(importances[i] * 100, 4))
        print(f"Corr Mean:{round(corr_mean[i], 3)}, Uncorr Mean:{round(uncorr_mean[i], 3)}")
        print(f"Corr Std:{round(corr_std[i], 3)}, Uncorr Std:{round(uncorr_std[i], 3)}")
        print("——————————————————————————————————————————————————————————————————")

if __name__ == "__main__":
    args = parse_args()
    main(args)
