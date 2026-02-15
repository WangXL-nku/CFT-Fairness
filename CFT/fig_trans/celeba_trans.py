import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

base_dir = '/media/data0/wzy/Craft/celeba_concept'
save_dir = '/media/data0/wzy/Craft/celeba_images'
# model_name_list = ['Heavy_Makeup-0', 'Blond_Hair-0', 'Heavy_Makeup-1', 'Blond_Hair-1']
model_name_list = ['Male-0', 'Male-1']
concept_num = 20

def img_trans(img):
    img -= img.min()
    img /= img.max()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

def get_crop_loc(crops_id, stride=51, patch_size=64):
    crops_id = crops_id % 16
    num_cols = 4

    # 计算行索引和列索引
    row_index = crops_id // num_cols
    col_index = crops_id % num_cols

    # 计算patch在原始图像中的起始位置
    x_start = col_index * stride
    y_start = row_index * stride

    # 计算patch在原始图像中的结束位置
    x_end = x_start + patch_size - 1
    y_end = y_start + patch_size - 1

    return (x_start, y_start, x_end, y_end)


def add_black_frame(image, crops_id, frame_width=2):

    # 创建一个可以用来绘制的ImageDraw对象
    draw = ImageDraw.Draw(image)
    position = get_crop_loc(crops_id)
    # 定义黑框的起点和终点
    x_start, y_start, x_end, y_end = position

    # 绘制黑框
    draw.rectangle([x_start, y_start, x_end, y_end], outline="white", width=frame_width)
    return image

for model_name in model_name_list:

    for i in range(concept_num):
        concept_path = os.path.join(base_dir, model_name, f'Concept{i}.npz')
        concept_array = np.load(concept_path)

        concept_dir = os.path.join(save_dir, model_name, f'Concept{i}')
        if not os.path.exists(concept_dir):
            os.makedirs(concept_dir)

        # 获取图片数组
        crops = concept_array['crops']
        imgs = concept_array['best_imgs']
        crops_ids = np.argsort(concept_array['importances'])[::-1][:crops.shape[0]]

        # 保存每张图片
        for j in range(crops.shape[0]):
            crop = crops[j]
            image = imgs[j]
            crops_id = crops_ids[j]
            image = image.transpose((1, 2, 0))

            # # 将图片数据转换为0-255范围
            crop = img_trans(crop)
            image = img_trans(image)
            # crop.show()   
            # image.show()
            add_black_frame(image, crops_id)
            # image.show()

            # 创建图片保存路径
            crop_path = os.path.join(concept_dir, f'{model_name}_crop_{i}_{j}.png')
            # img_path = os.path.join(concept_dir, f'{model_name}_img_{i}_{j}.png')
            img_crop_path = os.path.join(concept_dir, f'{model_name}_img_crop_{i}_{j}.png')

            # 保存图片
            crop.save(crop_path)
            # image.save(img_path)
            image.save(img_crop_path)
            # plt.imsave(crop_path, crop)
            # plt.imsave(img_path, image)
        print(model_name, f'Concept{i}', "completed!")