import os
import shutil
from PIL import Image

base_dir = '/media/data1/wzy/Craft/20/concept_images'
# model_name_list = ['Big_Nose-1', 'Heavy_Makeup-1', 'Male-1', 'Male-0']
model_name_list = ['Heavy_Makeup-1']
concept_num = 20
save_dir = '/media/data0/wzy/Craft/concept_images_trans'

for model_name in model_name_list:
    for i in range(concept_num):
        concept_dir = os.path.join(base_dir, model_name, f'Concept{i}')
        save_concept_dir = os.path.join(save_dir, model_name, f'Concept{i}')
        if not os.path.exists(save_concept_dir):
            os.makedirs(save_concept_dir)

        for j in range(20):
            crop_path = os.path.join(concept_dir, f'{model_name}_crop_{i}_{j}.png')
            img_crop_path = os.path.join(concept_dir, f'{model_name}_img_crop_{i}_{j}.png')

            # 读取图片
            try:
                with Image.open(crop_path) as img:
                    img.save(os.path.join(save_concept_dir, os.path.basename(crop_path)))
            except IOError:
                print(f"Cannot open {crop_path}")

            try:
                with Image.open(img_crop_path) as img:
                    img.save(os.path.join(save_concept_dir, os.path.basename(img_crop_path)))
            except IOError:
                print(f"Cannot open {img_crop_path}")