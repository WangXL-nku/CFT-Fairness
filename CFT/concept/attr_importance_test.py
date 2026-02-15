import numpy as np
import argparse
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

cuda_device = [0, 1, 2]
torch.cuda.set_device(cuda_device[0])  # 设置默认的GPU
attributes = ['', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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


class CelebADataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.img_dir = '/media/data0/wzy/dataset/CelebA/test'
        self.attr_path = os.path.join(args.base_save_path, 'dataset/CelebA/Anno/list_attr_celeba.txt')
        self.transform = transform
        with open(self.attr_path, 'r') as f:
            attr_lines = f.readlines()[2:]  # 跳过前两行
        self.image_paths = image_paths
        self.image_names = [path.split('/')[-1].split('.')[0] for path in image_paths]
        attr_labels = [attr_lines[int(image_name) - 1].strip().split()[args.target_attr] for
                       image_name in self.image_names]
        self.labels = [1 if int(attr_label) == 1 else 0 for attr_label in attr_labels]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Celeba Image Processing")
    parser.add_argument("-base_save_path", type=str, default='/media/data0/wzy/', help="Base save path for other files")
    parser.add_argument("-model_name", type=str, default='nf_resnet50.ra2_in1k')
    parser.add_argument("-layer", type=int, default=4, help="Directory to save figures")
    parser.add_argument("-target_attr", type=int, default=8)
    parser.add_argument("-target_attr_value", type=int, default=1)
    parser.add_argument("-image_num", type=int, default=3000)
    # parser.add_argument("-crops_num", type=int, default=20)
    parser.add_argument("-nb_crops", type=int, default=8, help="Number of crops to display")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-concept_num", type=int, default=20)
    parser.add_argument("-gpu", type=int, default=1)
    print(parser.parse_args())
    return parser.parse_args()


def test_acc(model, test_loader, verbose=0):
    running_corrects = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    accuracy = 1 - running_corrects.double() / len(test_loader.dataset)
    if verbose:
        print(f'Corr Accuracy: {accuracy:.4f}')
    else:
        print(f'Un Corr Accuracy: {accuracy:.4f}')
    return accuracy


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # 加载预训练的 ResNet50 模型
    model_path = os.path.join(args.base_save_path, 'Craft', 'models', 'celeba_models', attributes[args.target_attr],
                              'celeba_resnet50.pth')
    model = load_model(model_path, device)
    model = torch.nn.DataParallel(model, device_ids=cuda_device).cuda()
    concept_images_save_dir = os.path.join(args.base_save_path, 'Craft', 'concept_images_new', f"{attributes[args.target_attr]}-{args.target_attr_value}")
    if not os.path.exists(concept_images_save_dir):
        os.makedirs(concept_images_save_dir)
    corr_acc_list = []
    un_corr_acc_list = []
    for c_id in range(args.concept_num):
        concept_path = os.path.join(concept_images_save_dir, f'Concept{c_id}.npz')
        concept_array = np.load(concept_path)

        id_crops_u = concept_array['importances']
        sorted_indices = np.argsort(id_crops_u)[::-1]
        sorted_image_ids = [crop_id // 16 for crop_id in sorted_indices]
        img_paths = concept_array['img_paths']
        corr_imgs_ids = list(set(sorted_image_ids[:len(img_paths) // 2]))
        corr_imgs_paths = img_paths[corr_imgs_ids]
        un_corr_imgs_ids = list(set([i for i in range(len(img_paths))]) - set(corr_imgs_ids))
        un_corr_imgs_paths = img_paths[un_corr_imgs_ids]

        corr_dataset = CelebADataset(corr_imgs_paths, data_transforms)
        un_corr_dataset = CelebADataset(un_corr_imgs_paths, data_transforms)
        corr_loader = DataLoader(corr_dataset, batch_size=args.batch_size * len(cuda_device), shuffle=False)
        un_corr_loader = DataLoader(un_corr_dataset, batch_size=args.batch_size * len(cuda_device), shuffle=False)
        corr_acc = test_acc(model, corr_loader, verbose=1)
        un_corr_acc = test_acc(model, un_corr_loader, verbose=0)
        print("————————Concept", c_id, " has been completed!————————")
        corr_acc_list.append(round(corr_acc.cpu().numpy()*100, 2))
        un_corr_acc_list.append(round(un_corr_acc.cpu().numpy()*100, 2))
    print(corr_acc_list)
    print(un_corr_acc_list)



if __name__ == "__main__":
    args = parse_args()
    main(args)
