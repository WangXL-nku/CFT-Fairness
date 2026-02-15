import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

base_dir = '/media/data1/wzy/Craft/celeba_concept'
save_dir = '/media/data1/wzy/Craft/celeba_images'
# model_name_list = ['Heavy_Makeup-0', 'Heavy_Makeup-1', 'Blond_Hair-0', 'Blond_Hair-1']
model_name_list = ['Blond_Hair-0', 'Blond_Hair-1']
concept_num = 15

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

def gen_large_fig(img_list):
    # 创建一个空白的大图片，大小为512x512（因为8*64=512），颜色为黑色
    large_image = Image.new('RGB', (8 * 64, 8 * 64), 'black')

    # 遍历每张小图片
    for i in range(len(img_list)):
        # 计算小图片在大图片中的位置
        row = i // 8  # 行
        col = i % 8  # 列
        x = col * 64
        y = row * 64

        # 将小图片转换为PIL Image对象
        small_image = img_list[i]

        # 将小图片粘贴到大图片的相应位置
        large_image.paste(small_image, (x, y))

        # 如果不是最后一列，绘制右边的白线
        if col < 7:
            large_image.paste((255, 255, 255), (x + 64, 0, x + 65, 8 * 64))

        # 如果不是最后一行，绘制下面的白线
        if row < 7:
            large_image.paste((255, 255, 255), (0, y + 64, 8 * 64, y + 65))
    return large_image

def load_crops(model_name, concept_id, crop_num=64):
    concept_path = os.path.join(base_dir, str(concept_num), model_name, f'Concept{concept_id}.npz')
    concept_array = np.load(concept_path)

    concept_dir = os.path.join(base_dir, str(concept_num), model_name, f'Concept{concept_id}')
    if not os.path.exists(concept_dir):
        os.makedirs(concept_dir)

    # 获取图片数组
    crops = concept_array['crops']
    imgs = concept_array['best_imgs']
    crops_ids = np.argsort(concept_array['importances'])[::-1][:crops.shape[0]]
    crop_list = []
    image_list = []
    # 保存每张图片
    for j in range(crop_num):
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
        crop_list.append(crop)
        image_list.append(image)
    return crop_list, image_list

for model_name in model_name_list:

    for i in range(concept_num):
        crop_list, image_list = load_crops(model_name, i)
        large_img = gen_large_fig(crop_list)
        concept_img_save_dir = os.path.join(save_dir, str(concept_num), model_name)
        if not os.path.exists(concept_img_save_dir):
            os.makedirs(concept_img_save_dir)
        concept_img_save_path = os.path.join(concept_img_save_dir, f'Concept{i}.png')
        large_img.save(concept_img_save_path)
