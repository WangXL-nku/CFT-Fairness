import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

base_dir = '/media/data0/wzy/Craft/cifar-10_concept'
img_dir = '/media/data0/wzy/Craft/cifar-10_images'
model_name_list = [0.7, 0.95]
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

for model_name in model_name_list:

    for class_id in range(10):

        concept_path = os.path.join(base_dir, str(model_name), str(class_id), 'image.npy.npz')
        concept_array = np.load(concept_path)

        # 获取图片数组
        crops = concept_array['crops']
        crops_importances = concept_array['crop_importances']
        images = concept_array['images']
        concept_importance = concept_array['concept_images']

        for i in range(10):
            image_save_dir = os.path.join(img_dir, str(model_name), str(class_id), f"Concept{i}")
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            concept_crops_u = crops_importances[:, i]

            sorted_indices = np.argsort(concept_crops_u)[::-1]
            best_crops_ids = sorted_indices[:save_fig_num]
            best_crops = crops[best_crops_ids]
            best_img_ids = [crop_id // 9 for crop_id in best_crops_ids]
            best_imgs = images[best_img_ids]

            # 保存每张图片
            for j in range(best_crops.shape[0]):
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
                # image.show()

                # 创建图片保存路径
                crop_path = os.path.join(image_save_dir, f'crop_{model_name}_{class_id}_Concept{i}_{j}.png')
                img_crop_path = os.path.join(image_save_dir, f'img_crop_{model_name}_{class_id}_Concept{i}_{j}.png')
                # img_path = os.path.join(concept_dir, f'{model_name}_img_{i}_{j}.png')

                # 保存图片
                crop.save(crop_path)
                # image.save(img_path)
                image.save(img_crop_path)
                # plt.imsave(crop_path, crop)
                # plt.imsave(img_path, image)
            print(model_name, f"class{class_id}", f'Concept{i}', f"importance:{round(concept_importance[i]*100, 4)}", "completed!")