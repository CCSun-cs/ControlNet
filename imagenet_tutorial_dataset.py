import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet_lable_original.json', 'rt') as f:
        #
        # with open("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_noprompt_new.json", 'rt') as f:##sccchange
        # with open("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_prompt_new.json", 'rt') as f:##sccchange
        # with open("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_benign_new.json", 'rt') as f:##sccchange
        with open("/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_source_adv_source_original_new.json", 'rt') as f:##sccchange

            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']##origin
        ###将source和target都当做control，看看能不能生成相同类别的图片
        # source_filename = item['target']
        target_filename = item['target']
        prompt = item['prompt']                ## this is the text input, 
        gt_key = item['ground_truth']          ##target flipped label

        source = cv2.imread( source_filename)  ##condition image
        target = cv2.imread( target_filename)  ## ground truth image

        # Do not forget that OpenCV read images in BGR order.
        try:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        except:
            print(source_filename,target_filename)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        source = resize_image(source, 512)
        target = resize_image(target, 512)

        # return dict(jpg=target, txt=prompt, hint=source)
        return dict(jpg=target, txt=prompt, hint=source,gt_txt=gt_key)


##########################
# import os
# import json
# from tqdm import tqdm
# # # ##############iamgenet
# # prompt_cifar10 = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet.json"
# # prompt_imagenet = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet_noprompt_new.json"
# prompt_imagenet = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_source_adv_source_original_new.json"

# # path_s = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/source"
# # path_t = "/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/target"

# path_s = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_adv"
# path_t = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_original"
# images = os.listdir(path_s)
# print("here")
# with open(prompt_imagenet, "w") as json_lines_file:
#     for img in tqdm(images):
#     #dog__cat__44538.png
#         # print(img)
#         img_dict = {}
#         img_dict["source"] = os.path.join(path_s,img)
#         img_dict["target"] = os.path.join(path_t,img)
#         label_s = img.split("__")[0]
#         label_t = img.split("__")[1]
#         # print(label_s,label_t)
#         # img_dict["prompt"] = "An image of "+label_s+" is classified as "+label_t
#         # img_dict["prompt"] = "An image of "+label_s
#         # img_dict["prompt"] = "An image of "+label_t
#         img_dict["prompt"] = "An image of "+label_s
#         # img_dict["prompt"] = " "
#         # img_dict["ground_truth"] = label_s+"\t"+label_t
#         # img_dict["ground_truth"] = label_t+"\t"+label_t
#         img_dict["ground_truth"] = label_s+"\t"+label_s
#         json.dump(img_dict, json_lines_file)
#         json_lines_file.write("\n")
