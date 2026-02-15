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
parser.add_argument('-img_base_dir', type=str, default='/concept_images', help='The base directory for image files.')
parser.add_argument('-model_id', type=int, default=0, help='The index of the model to use.')
parser.add_argument('-img_num', type=int, default=20 , help='The number of images to process.')
parser.add_argument('-gpu', type=int, default=0, help='The number of images to process.')
# 解析参数
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
model_name_list = ['Big_Nose-1', 'Heavy_Makeup-1', 'Male-1', 'Male-0']


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
    # print(image_concent)
    # prompt =  {"type": "text", "text": "请仔细查看这些图片，并仅识别图中白框内的内容。你的任务是描述白框内的物体、特征或概念，并且只关注白框内的区域。例如，如果白框内显示的是一只眼睛，请直接回答：“白框内是一只眼睛。” 忽略白框外的所有内容，确保你的描述只包含白框内识别的内容。"}
    # prompt =  {"type": "text", "text": "请仔细查看这些图片，并总结这些图中白框内的内容。你的任务是描述白框内的物体、特征或概念，并且只关注白框内的区域，可能的概念特征包括：额头、眉毛、眼睛、牙齿、睫毛、耳朵、脖子、嘴巴、鼻子、嘴唇、头发、耳钉、帽子、眼镜、项链、领带、领结、衬衫、口红、微笑、亲切、attractive、金黄的头发、黑发、红发、直发、波浪卷发、宽大的鼻子、绿色的背景、黑色的背景、条纹背景、图片背景、化了妆的面容、左边眼睛、侧脸、男性的正脸等。例如，如果你认为白框区域都与眼睛相关，请直接回答：“白框内是一只眼睛。” 忽略白框外的所有内容，确保你的描述只包含白框内识别的内容。"}
    # prompt =  {"type": "text", "text": "请仔细查看这些图片，并总结这些图中白框内的内容的共同点。你的任务是描述白框内的物体、特征或概念，并且只关注白框内的区域，忽略白框外的所有内容，确保你的描述只包含白框内识别的内容。可能的概念特征包括：头顶、手指、手、下巴、脸颊、额头、眉毛、眼睛、牙齿、睫毛、耳朵、脖子、嘴巴、下颚、脖颈、鼻子、嘴唇、肩膀、头发、耳钉、帽子、眼镜、衣服、项链、领带、领结、衬衫、口红、微笑、亲切、attractive、秃顶、年迈、年轻、光滑的皮肤、褶皱的皮肤、金黄的头发、黑发、红发、直发、波浪卷发、宽大的鼻子、绿色的背景、黑色的背景、条纹背景、图片背景、化了妆的面容、左边眼睛、侧脸、男性的正脸等。"}
    '''
    log 3
    '''
    # prompt = {"type":
    #               "text",
    #           "text":
    #               "-Role:白框内语义概念识别任务"
    #               "-Background:用户会给你一组人脸照片，需要你识别每组照片中白框内部分所对应的语义特征概念"
    #               "Goals:你需要总结这些图白框中内容的共同点，并输出一组图片中白框内部分可能对应的语义特征概念"
    #               "-Constrains:你应当只关注白框内的区域，忽略白框外所有内容，确保你的描述只包含白框内识别的内容。同时应当总结一组图片中白框内部分的共性，推测这一组图片白框内部分的内容而不是基于单个图片做识别"
    #               "-OutputFormat: 用文本输出一组图片白框内部分可能代表的语义"
    #               "-Workflow: 1. 识别每张图片白框内部分可能包含的语义概念特征;2. 总结这组图片白框内部分可能包含的语义特征概念，去除由单个图片引起的特异性差异，总结一组图片白框部分语义的共性; 3. 输出这组图片白框内部分对应的语义特征概念"
    #               "-可能的语义特征概念：头顶、手指、手、下巴、脸颊、额头、眉毛、眼睛、牙齿、睫毛、耳朵、脖子、嘴巴、下颚、脖颈、鼻子、嘴唇、肩膀、头发、耳钉、帽子、眼镜、衣服、项链、领带、领结、衬衫、口红、微笑、亲切、attractive、秃顶、年迈、年轻、光滑的皮肤、褶皱的皮肤、金黄的头发、黑发、红发、直发、波浪卷发、宽大的鼻子、绿色的背景、黑色的背景、条纹背景、图片背景、与人脸无关的图片背景、化了妆的面容、左边眼睛、侧脸、男性的正脸、男性的头顶、女性的侧脸、年轻男性的脖颈等等。"}

    prompt = {"type":
                  "text",
              "text":
                  "-Role:一组人脸图像白框内语义概念识别任务"
                  "-Background:用户会给你一组人脸照片，每张照片中的特定区域由白框圈出，并且一组图片中被白框圈出的部分往往具有相似的语义概念，需要你分析白框区域与图像中人脸的相对位置关系，识别与白框区域中出现过的特征概念，总结一组图像中白框区域的共性语义特征。在人脸图片中，包含与人脸相关的语义特征和与人脸无关的语义特征，举例如下：[人脸相关的语义特征：鼻子（人脸正脸区域）、眼睛（人脸正脸区域，人脸上半部分）、嘴巴（人脸正脸区域，人脸下半部分）、长发（人脸头顶区域，人脸侧边区域）、短发（人脸头顶区域）、波浪头发（人脸头顶区域，人脸侧边区域）、直发（人脸头顶区域，人脸侧边区域）、下巴（人脸下半部分）、下颚（人脸下半部分）、脖颈（人脸下半部分、人脸外侧区域）、睫毛（人脸正脸区域，人脸上半部分）、耳朵（人脸正脸区域、人脸侧边区域）、额头（人脸正脸区域，人脸上半部分）、脸颊（人脸正脸区域，人脸侧边区域）、耳环（人脸外侧区域）、眼镜（人脸正脸区域，人脸上半部分）等等; 人脸无关的语义特征：衣服（与人脸无关的背景区域）、背景（与人脸无关的背景区域）、手指（人脸外侧区域）、肩膀（人脸下半部分、与人脸无关的背景区域）等等;]"
                  "-Goals:识别与一组图像中白框区域最相关的语义特征，包括白框区域相对人脸的位置特征、白框区域中出现过语义特征、一组图片中白框区域的共性特征，判断白框内的语义特征是否是人脸无关的语义特征。"
                  "-Constrains:"
                  " a.你应当精确关注白框内的区域，忽略白框外所有内容，确保你的描述只包含白框内识别的内容，例如图片出现了女性的面部特征，如耳朵、鼻子、眼睛等，但白框区域位置集中在人脸图像头顶，仅仅包括了女性的头发特征，你不应当将白框外的耳朵、鼻子、眼睛等特征作为识别到的语义特征输出。"
                  " b.你的识别结果应当满足人脸图像中可能存在的语义概念的物理位置对应关系，例如：图像中白框区域集中在人脸外部，那么白框内区域不应该被识别为鼻子、眼睛、嘴巴等人脸正脸区域的语义特征;如果图像中白框区域集中在人脸上半部分区域，则不应该出现嘴巴特征，因为嘴巴往往在人脸下半部分区域。"
                  " c.你应当重点识别可能与人脸无关的语义特征，如果一组图片中白框区域大概率与人脸特征无关，你需要在输出中表明白框区域可能相关的概念是人脸无关的语义概念。"
                  "-Workflow: "
                  " 1. 识别每张图片白框区域对应人脸的位置关系，判断白框区域相较于人脸是否有较强的物理聚集关系，例如：图像中白框区域集中在人侧脸区域；图像中白框区域集中在人脸上半部分区域；图像内白框区域没有固定的位置信息，在人脸外侧、人脸上方、背景区域都出现过。"
                  " 2. 基于白框相对于人脸的位置关系，结合原图识别白框区域对应的简单语义特征，首先识别白框内包括的部分是否与人脸语义特征相关，如果是无关特征，则直接跳到步骤5，开始总结白框内语义的共性。当白框区域集中在图片中人脸外的区域时，白框中往往是人脸无关的语义特征。"
                  " 3. 如果白框内部分是与人脸相关的语义特征，则基于白框相对于人脸的位置关系，结合原图识别白框区域对应的人脸相关的语义特征"
                  " 4. 原图和白框内相关的人脸相关的语义特征，进一步识别白框区域对应的复合语义特征，如：微笑的嘴巴、带着笑意的眼睛、愤怒的眼神、秃顶的男性正脸、年轻的脸颊、光滑的面部皮肤、褶皱的面部皮肤、有吸引力的面容、金黄的头发、黑发、红色的头发、宽大的鼻子、绿色的背景、黑色的背景、条纹背景、化了妆的面容、亲切的表情等等; "
                  " 5. 总结这组图片白框内部分可能包含的语义特征概念，去除由单个图片引起的特异性差异，总结一组图片白框部分语义的共性; "
                  " 6. 输出这组图片白框内部分对应的语义特征概念。"
                  "-Note: 图片中白框所在的区域有可能与人脸特征无关或关系很小，仅仅包含背景区域或是一些无关特征，如果是与人脸无关的区域，你需要将其识别为无关特征，例如：背景区域、条纹图案、手、胳膊、脖颈、肩膀等人的身体特征等等。"
                  "-OutputFormat: 首先描述这组图像中白框区域相对于人脸的位置关系，再这组图片白框内区域出现过的语义概念，最后总结这组图像白框区域最有可能相关的语义概念"
                  "-Examples: "
                  "    1. 与这组图片中白框区域位置集中在人脸侧方区域，可能相关的语义概包括耳环、人的侧脸区域、耳朵、头发、眼角等等，这组图像白框区域最有可能相关的语义概念是侧脸视角下的耳朵。"
                  "    2. 与这组图片中白框区域位置集中在人脸正脸区域，可能相关的语义概包括化了妆的眼睛、化了妆的眉毛、窄小的鼻子、涂了口红的嘴唇、化了妆的脸颊等等,这组图像白框区域最有可能相关的语义概念是女性的正脸。"
                  "    3. 与这组图片中白框区域位置集中在人脸头顶区域和人脸侧方区域，可能相关的语义概包括头顶黑色的直发、头顶黑色的卷发、头顶的背景区域等等,这组图像白框区域最有可能相关的语义概念是头顶的头发。"
                  "    4. 与这组图片中白框区域位置集中在人脸侧方区域，可能相关的语义概包括头顶脸颊旁的金色波浪卷发、女性的耳朵、女性的脸颊、人脸侧方向的背景特征等等,这组图像白框区域最有可能相关的语义概念是女性侧脸区域的波浪卷发。"
                  "    5. 与这组图片中白框区域位置没有集中在特定区域，可能相关的语义概包括绿色背景、条纹背景等等,这组图像白框区域最有可能相关的语义概念是与人脸无关的图片绿色背景区域。"}
    image_concent.append(prompt)
    if verbose:
        print(image_concent)
    return image_concent


for i in [1, 16,17, 18]:
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