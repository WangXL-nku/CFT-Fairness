import time
import os
import numpy as np
import argparse
import ast
from openai import OpenAI
import base64
import random

attributes = ['', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

choosed_attr_list = ['Attractive', 'Big_Nose', 'Blond_Hair', 'Heavy_Makeup', 'Male', 'Smiling']
choosed_attr_id = [3, 8, 10, 19, 21, 32]


def parse_args():
    parser = argparse.ArgumentParser(description="Celeba Image Processing")
    parser.add_argument("-base_save_path", type=str, default='/media/data0/wzy/', help="Base save path for other files")
    parser.add_argument("-image_num", type=int, default=30)
    # parser.add_argument("-crops_num", type=int, default=20)
    print(parser.parse_args())
    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gen_query(image_path, exp_image_paths, exp_image_attr_lists, verbose=0):
    image_concent = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
        } for image_path in exp_image_paths
    ]
    example_str = f"-Examples：对于上面给出{len(exp_image_paths)}张图像，按照顺序其对应的属性标签分别为："
    for i in range(len(exp_image_attr_lists)):
        example_str += f"[{', '.join(map(str, exp_image_attr_lists[i]))}], "
    prompt = {"type":
                  "text",
              "text":
                  "-Role: 人脸图像特征标注任务。"
                  "-Background: 人脸图像需要标注的特征有：['Attractive', 'Big_Nose', 'Blond_Hair', 'Heavy_Makeup', 'Male', 'Smiling']。具体特征含义为：{Attractive：图像中人脸是否具有吸引力；Big_Nose：图像中人脸鼻子是否较大的鼻子；Blond_Hair：图像中人脸是否是金发；Heavy_Makeup：图像中人脸是否有明显化妆；Male：图像中人脸的性别是否为男性；Smiling：图像中人脸是否在微笑。}"
                  "-Goals：基于样例图片及其对应特征的标签，输出下面给出的图像的特征标签。"
                  "-Workflow："
                  "1. 识别测试图像在每个特征下的标签"
                  "2. 按照['Attractive', 'Big_Nose', 'Blond_Hair', 'Heavy_Makeup', 'Male', 'Smiling']的顺序将标签值转换为1和-1"
                  "3. 输出图像特征标签"
                  "-OutputFormat：对每一个特征，用1和-1表示图像中人脸是否满足对应特征，并将所有的特征用一个数组的形式输出，一张图像被输出为[1,-1,1,1,-1,-1]，意味着识别这张图像的人脸是[具有吸引力的，鼻子并非较大的鼻子，金发，有明显化妆，女性，并未在微笑]。"
                  "" + example_str
              }
    test_case = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
    }
    image_concent.append(prompt)
    image_concent.append(test_case)
    if verbose:
        print(image_concent)
    return image_concent


def load_path_list(args, example=0):
    if example:
        set_str = '0'
        target_num = args.image_num
        add = random.randint(0, 2000)
        test_images_dir = os.path.join(args.base_save_path, 'dataset/CelebA/train')
    else:
        set_str = '2'
        target_num = args.image_num
        add = random.randint(0, 2000)
        test_images_dir = os.path.join(args.base_save_path, 'dataset/CelebA/test')
    attr_file_path = os.path.join(args.base_save_path, 'dataset/CelebA/Anno/list_attr_celeba.txt')
    partition_file_path = os.path.join(args.base_save_path, 'dataset/CelebA/Eval/list_eval_partition.txt')

    with open(partition_file_path, 'r') as file:
        eval_lines = file.readlines()

    with open(attr_file_path, 'r') as file:
        lines = file.readlines()[2:]

    assert len(eval_lines) == len(lines)

    target_image_paths = []
    image_attr_lists = []

    for i in range(add, len(eval_lines)):
        image_set = eval_lines[i].strip().split()[1]
        attr_line_split = lines[i].strip().split()
        img_name = attr_line_split[0]
        if image_set == set_str:
            image_path = os.path.join(test_images_dir, img_name)
            target_image_paths.append(image_path)
            image_attr_lists.append(np.array(attr_line_split)[choosed_attr_id].astype(int))
            if len(target_image_paths) == target_num:
                break
    return target_image_paths, image_attr_lists


def main(args):
    target_image_paths, image_attr_lists = load_path_list(args)
    exp_image_paths, exp_image_attr_lists = load_path_list(args, example=1)
    client = OpenAI(
        api_key='sk-2a24ccb5c69544cea3686fb2b96b5f18',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    image_acc_list = []
    for i in range(len(target_image_paths)):
        query_content = gen_query(target_image_paths[i], exp_image_paths, exp_image_attr_lists)
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user",
                "content": query_content
            }]
        )
        result = completion.choices[0].message.content
        result = np.array(ast.literal_eval(result))
        image_acc = result == image_attr_lists[i]
        image_acc_list.append(image_acc)
        print(image_acc)
        time.sleep(7)
    image_acc_list = np.array(image_acc_list)
    attr_acc = image_acc_list.sum(axis=0) / image_acc_list.shape[0]
    print(attr_acc)


if __name__ == "__main__":
    args = parse_args()
    main(args)
