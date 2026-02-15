#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from openai import OpenAI
import base64
import json

torch.manual_seed(1234)

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数
parser.add_argument('-img_base_dir', type=str, default='/media/data0/wzy/Craft/concept_images_new', help='The base directory for image files.')
parser.add_argument('-model_id', type=int, default=4, help='The index of the model to use.')
parser.add_argument('-img_num', type=int, default=20 , help='The number of images to process.')
parser.add_argument('-gpu', type=int, default=0, help='The number of images to process.')
# 解析参数
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
model_name_list = ['Big_Nose-1', 'Heavy_Makeup-1', 'Male-1', 'Male-0', 'Blond_Hair-1']


concept_num = 20

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_concept_img_path(concept_id):
    concept_path = os.path.join(args.img_base_dir, model_name_list[args.model_id], f'Concept{concept_id}')
    concept_image_crop_path_list = [os.path.join(concept_path, f'{model_name_list[args.model_id]}_img_crop_{concept_id}_{j}.png') for j in range(args.img_num)]
    concept_crop_path_list = [os.path.join(concept_path, f'{model_name_list[args.model_id]}_crop_{concept_id}_{j}.png') for j in range(args.img_num)]
    concept_image_path_list = [os.path.join(concept_path, f'{model_name_list[args.model_id]}_img_{concept_id}_{j}.png') for j in range(args.img_num)]
    return concept_crop_path_list, concept_image_path_list, concept_image_crop_path_list


def gen_query(concept_image_crop_path_list, verbose=0):
    image_concent = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"}
        } for image_path in concept_image_crop_path_list
    ]
    prompt = {"type":
                  "text",
              "text":
                  "-Role:识别一组图片白框中的内容是否明显有性别特征"
                  "-Background:用户会给你一组人脸照片，每张照片中的特定区域由白框圈出，并且一组图片中被白框圈出的部分往往具有相似的语义概念，需要你评估白框区域中的内容是否与男性或女性特征相关"
                  "-Constrains: "
                  "1. 你应当精确关注白框内的区域，忽略白框外所有内容，确保你的描述只包含白框内识别的内容，例如图片出现了女性的面部特征，如耳朵、鼻子、眼睛等，但白框区域位置集中在人脸图像头顶，仅仅包括了背景区域，则你不应该将其输出为具有女性特征；"
                  "2. 你应当总结一组图片白框内容的共性特征，如果一组图像中既有男性特征，又有女性特征，并且没有明显的倾向，那么你不应当将其识别为仅仅与男性或女性单个性别相关"
                  "-Workflow: "
                  " 1. 识别每张图片白框中内容对应的可能的语义，判断是否可能与男性或女性特征相关。"
                  " 2. 总结这组图片白框内部分可能包含的语义特征概念，去除由单个图片引起的特异性差异，总结一组图片白框部分语义的共性是否与男性或女性特征相关; "
                  " 3. 输出这组图片白框内部分对应的语义特征概念是否与男性或女性特征相关，如果是，与哪些男性或女性特征相关。"}
    #
    # prompt = {"type":
    #               "text",
    #           "text":
    #               "-Role:识别一组图片是否包含女性特征"
    #               "-Background:用户会给你一组人脸照片，每张照片中的特定区域由白框圈出，并且一组图片中被白框圈出的部分往往具有相似的语义概念，需要你评估白框区域中的内容是否与女性人脸特征相关"
    #               "-Constrains: 你应当精确关注白框内的区域，忽略白框外所有内容，确保你的描述只包含白框内识别的内容，例如图片出现了女性的面部特征，如耳朵、鼻子、眼睛等，但白框区域位置集中在人脸图像头顶，仅仅包括了背景区域，则你不应该将其输出为具有女性特征"
    #               "-Workflow: "
    #               " 1. 识别每张图片白框中内容对应的可能的语义，判断是否可能与女性特征相关。"
    #               " 2. 总结这组图片白框内部分可能包含的语义特征概念，去除由单个图片引起的特异性差异，总结一组图片白框部分语义的共性是否与女性特征相关; "
    #               " 3. 输出这组图片白框内部分对应的语义特征概念是否与女性特征相关，如果是，与哪些女性特征相关。"
    #               "-Note: 图片中白框所在的区域与图片的所有内容并不等同，可能图片均为女性人脸图像，但白框区域集中在非女性特征区域。"}
    image_concent.append(prompt)
    if verbose:
        print(image_concent)
    return image_concent


for i in range(20):
    client = OpenAI(
        api_key='sk-2a24ccb5c69544cea3686fb2b96b5f18',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    concept_crop_path_list, concept_image_path_list, concept_image_crop_path_list = get_concept_img_path(i)
    print(f"——————————————Concept{i}————————————————")
    query_content = gen_query(concept_image_crop_path_list)
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[{
                "role": "user",
                "content": query_content
            }]
    )
    result = completion.choices[0].message.content
    print(result)