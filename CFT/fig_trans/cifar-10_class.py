import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

base_dir = '/media/data0/wzy/Craft/cifar-10_concept'
img_dir = '/media/data0/wzy/Craft/cifar-10_images'
model_name_list = [0.95]
save_fig_num = 50


def img_trans(img):
    img -= img.min()
    img /= img.max()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def get_crop_loc(crops_id, stride=8, patch_size=10):
    crops_id = crops_id % 9
    num_cols = 3

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


def add_black_frame(image, crops_id, frame_width=1):
    # 创建一个可以用来绘制的ImageDraw对象
    draw = ImageDraw.Draw(image)
    position = get_crop_loc(crops_id)
    # 定义黑框的起点和终点
    x_start, y_start, x_end, y_end = position

    # 绘制黑框
    draw.rectangle([x_start, y_start, x_end, y_end], outline="white", width=frame_width)
    return image


def gen_large_fig(img_list, base_size=10, col_num=3):
    # 创建一个空白的大图片，大小为512x512（因为8*64=512），颜色为黑色
    large_image = Image.new('RGB', (col_num * base_size, col_num * base_size), 'black')

    # 遍历每张小图片
    for i in range(len(img_list)):
        # 计算小图片在大图片中的位置
        row = i // col_num  # 行
        col = i % col_num  # 列
        x = col * base_size
        y = row * base_size

        # 将小图片转换为PIL Image对象
        small_image = img_list[i]

        # 将小图片粘贴到大图片的相应位置
        large_image.paste(small_image, (x, y))

        # 如果不是最后一列，绘制右边的白线
        if col < col_num-1:
            large_image.paste((255, 255, 255), (x + base_size, 0, x + base_size+1, col_num * base_size))

        # 如果不是最后一行，绘制下面的白线
        if row < col_num-1:
            large_image.paste((255, 255, 255), (0, y + base_size, col_num * base_size, y + base_size+1))
    return large_image


def load_crops(crops, crops_importances, images, concept_id, crop_num=64):
    concept_crops_u = crops_importances[:, concept_id]

    sorted_indices = np.argsort(concept_crops_u)[::-1]
    best_crops_ids = sorted_indices[:crop_num]
    best_crops = crops[best_crops_ids]
    best_img_ids = [crop_id // 9 for crop_id in best_crops_ids]
    best_imgs = images[best_img_ids]

    crop_list = []
    image_list = []
    # 保存每张图片
    for j in range(crop_num):
        crop = best_crops[j]
        image = best_imgs[j]
        crops_id = best_crops_ids[j]
        image = image.transpose((1, 2, 0))

        # # 将图片数据转换为0-255范围
        crop = img_trans(crop)
        image = img_trans(image)
        # crop.show()
        # image.show()
        add_black_frame(image, crops_id)
        crop_list.append(crop)
        image_list.append(image)
    return crop_list, image_list


for model_name in model_name_list:

    for class_id in range(10):

        concept_path = os.path.join(base_dir, str(model_name), str(class_id), 'image.npy.npz')
        concept_array = np.load(concept_path)

        # 获取图片数组
        crops = concept_array['crops']
        crops_importances = concept_array['crop_importances']
        images = concept_array['images']
        concept_importance = concept_array['concept_images']
        concept_img_list = []
        concept_img_save_path = os.path.join(img_dir, str(model_name), f"{class_id}.png")
        for i in range(9):

            crop_list, image_list = load_crops(crops, crops_importances, images, i, crop_num=9)

            large_img = gen_large_fig(crop_list, base_size=10, col_num=3)
            concept_img_list.append(large_img)
        large_img = gen_large_fig(concept_img_list, base_size=10*3, col_num=3)
        large_img.save(concept_img_save_path)
