import os
import json
# # ##############iamgenet
# # prompt_cifar10 = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet.json"
# prompt_imagenet = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet_noprompt_new.json"

# path = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/source"
# path_t = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/target"
# images = os.listdir(path)
# with open(prompt_imagenet, "w") as json_lines_file:
#     for img in images:
#     #dog__cat__44538.png
#         print(img)
#         img_dict = {}
#         img_dict["source"] = os.path.join(path,img)
#         img_dict["target"] = os.path.join(path_t,img)
#         label_s = img.split("__")[0]
#         label_t = img.split("__")[1]
#         # img_dict["prompt"] = "An image of "+label_s+" is classified as "+label_t
#         # img_dict["prompt"] = "An image of "+label_s
#         img_dict["prompt"] = " "
#         img_dict["ground_truth"] = label_s+"\t"+label_t
#         json.dump(img_dict, json_lines_file)
#         json_lines_file.write("\n")


# ##############cifar10
# prompt_cifar10 = "/egr/research-optml/sunchan5/AEG/dataset/cifar10_final/train_controlnet/prompt_cifar10_noprompt.json"

# path = "/egr/research-optml/sunchan5/AEG/dataset/cifar10_final/train_controlnet/source"
# path_t = "/egr/research-optml/sunchan5/AEG/dataset/cifar10_final/train_controlnet/target"
# images = os.listdir(path)
# with open(prompt_cifar10, "w") as json_lines_file:
#     for img in images:
#     #dog__cat__44538.png
#         print(img)
#         img_dict = {}
#         img_dict["source"] = os.path.join(path,img)
#         img_dict["target"] = os.path.join(path_t,img)
#         label_s = img.split("__")[0]
#         label_t = img.split("__")[1]
#         # img_dict["prompt"] = "An image of "+label_s+" is classified as "+label_t
#         # img_dict["prompt"] = "An image of "+label_s
#         img_dict["prompt"] = " "
#         # img_dict["original_label"] = label_s
#         img_dict["ground_truth"] = label_s+"\t"+label_t
#         json.dump(img_dict, json_lines_file)
#         json_lines_file.write("\n")

##############
# import config

# import cv2
# import einops
# import gradio as gr
# import numpy as np
# import torch
# import random

# from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
# from annotator.canny import CannyDetector
# from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler



# source_filename = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/source/pot__cauliflower__20660.JPEG"
# source = cv2.imread( source_filename)
        
# # Do not forget that OpenCV read images in BGR order.
# source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

# H1, W1, C1 = source.shape
# print(H1,W1,C1)
# # Normalize source images to [0, 1].
# source = source.astype(np.float32) / 255.0

# image_resolution = 512
# img = resize_image(source, image_resolution)
# H, W, C = img.shape
# print(H,W,C)



###1098477

# import torch

# # 加载 checkpoint 文件
# resume_path = '/egr/research-optml/sunchan5/ControlNet/checkpoints_imagenet_noprompt_new/epoch=3-loss=0.00-step=52199.ckpt'
# # 指定加载的 CUDA 设备
# device = torch.device("cuda:1")  # 选择要加载到的 CUDA 设备

# # 加载 checkpoint 文件并将模型加载到指定的设备
# checkpoint = torch.load(resume_path, map_location=device)

# # 获取保存的 epoch 值
# epoch = checkpoint['epoch']

# print(f'The model was trained for {epoch} epochs.')

import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm
source_filename = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_adv"
target_filename = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_original"

# # shutil.copy(source_filename,"/egr/research-optml/sunchan5/1.JPEG")
# shutil.copy(target_filename,"/egr/research-optml/sunchan5/2.JPEG")

f1 = os.listdir("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_adv")
f2 = os.listdir("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_original")
print(len(f1),len(f2)) #1063194
count=0
for f in tqdm(f1):
    source = cv2.imread( os.path.join(source_filename,f))  ##condition image
    # print(source.shape)
    target = cv2.imread( os.path.join(target_filename,f))  ##condition image
    # print(f,source.shape,target.shape)
    if source.shape!=target.shape:
        print(f)
#     file_path =  os.path.join(target_filename,f)
#     if not os.path.exists(file_path):
#         print(f)
#         count+=1
# print(count)


# source_filename = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target/African_crocodile__mosquito_net__10103.JPEG"
# target_filename = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target3/African_crocodile__mosquito_net__10103.JPEG"

# # /egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target/African_crocodile__mosquito_net__10103.JPEG /egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target3/African_crocodile__mosquito_net__10103.JPEG

# source = cv2.imread( source_filename)  ##condition image
# target = cv2.imread( target_filename)  ## ground truth image

# print(source.shape)
# print(target.shape)

# # Do not forget that OpenCV read images in BGR order.
# try:
#     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# except:
#     print(source_filename,target_filename)
# # Normalize source images to [0, 1].
# print()
# source = source.astype(np.float32) / 255.0
# # Normalize target images to [-1, 1].
# target = (target.astype(np.float32) / 127.5) - 1.0