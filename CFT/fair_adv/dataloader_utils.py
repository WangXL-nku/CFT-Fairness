import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class CelebADataset(Dataset):
    def __init__(self, base_path, target_attr=10, target_attr_value=0, image_num=3000, split='test', transform=None):
        self.base_path = base_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5063, 0.4258, 0.3832], [0.2669, 0.2456, 0.2413])
        ])

        attr_file = os.path.join(base_path, 'Anno/list_attr_celeba.txt')
        partition_file = os.path.join(base_path, 'Eval/list_eval_partition.txt')
        images_dir = os.path.join(base_path, split)

        target_attr_v = 1 if target_attr_value == 1 else -1
        split_code = '2' if split == 'test' else ('1' if split == 'val' else '0')

        with open(partition_file, 'r') as f:
            eval_lines = f.readlines()
        with open(attr_file, 'r') as f:
            attr_lines = f.readlines()[2:]

        self.image_paths = []
        for eval_line, attr_line in zip(eval_lines, attr_lines):
            if len(self.image_paths) >= image_num:
                break

            img_set = eval_line.strip().split()[1]
            attr_parts = attr_line.strip().split()
            img_name = attr_parts[0]
            target_label = int(attr_parts[target_attr])

            if img_set == split_code and target_label == target_attr_v:
                img_path = os.path.join(images_dir, img_name)
                self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.image_paths[idx]


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_task='gender'):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_task = target_task
        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(root_dir):
            if img_name.endswith('.jpg'):
                parts = img_name.split('_')
                if len(parts) >= 3:
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                        race = int(parts[2])

                        self.image_paths.append(os.path.join(root_dir, img_name))

                        if target_task == 'gender':
                            self.labels.append(gender)
                        elif target_task == 'race':
                            self.labels.append(race)
                        elif target_task == 'age':
                            self.labels.append(age)
                    except:
                        continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.image_paths[idx]


class LFWImageDataset(Dataset):
    def __init__(self, min_faces_per_person=70, resize=0.4, transform=None):
        from sklearn.datasets import fetch_lfw_people
        lfw_data = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)

        self.images = lfw_data.images
        self.targets = lfw_data.target
        self.target_names = lfw_data.target_names

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        img_pil = Image.fromarray(img).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img_pil)
        return img_tensor, f"person_{self.targets[idx]}"


class CIFAR10SDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label2=2, label3=3):
        self.root_dir = os.path.join(root_dir, split)
        self.label2 = label2
        self.label3 = label3
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        self.image_paths = []
        self.labels = []

        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                class_idx = int(class_dir)
                if class_idx == label2 or class_idx == label3:
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(0 if class_idx == label2 else 1)

        self._apply_bias()

    def _apply_bias(self):
        import random
        modified_images = []

        for img_path, label in zip(self.image_paths, self.labels):
            img = Image.open(img_path).convert('RGB')
            if label == 0:
                if random.random() < 0.95:
                    img = img.convert('L').convert('RGB')
            else:
                if random.random() < 0.05:
                    img = img.convert('L').convert('RGB')

            modified_images.append(img)

        self.images = modified_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def create_dataset(dataset_name, **kwargs):
    if dataset_name == 'celeba':
        return CelebADataset(**kwargs)
    elif dataset_name == 'utkface':
        return UTKFaceDataset(**kwargs)
    elif dataset_name == 'lfw':
        return LFWImageDataset(**kwargs)
    elif dataset_name == 'cifar10s':
        return CIFAR10SDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_images_from_paths(image_paths, transform=None):
    images = []
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform = transform or default_transform

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)

    return torch.stack(images, 0)