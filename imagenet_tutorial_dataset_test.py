from imagenet_tutorial_dataset import MyDataset
import json
import cv2
import numpy as np
from annotator.util import resize_image, HWC3
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
dataset = MyDataset()
print(len(dataset))

# seed=42
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True,worker_init_fn=lambda x: random.seed(seed))

# for batch in tqdm(dataloader):
#     print("****")
data = []
with open('/egr/research-optml/sunchan5/AEG/dataset/ImageNet_final/train_controlnet/prompt_imagenet_noprompt.json', 'rt') as f:
    for line in f:
        data.append(json.loads(line))

for idx in tqdm(range(len(data))):
    item = data[idx]
    source_filename = item['source']
    target_filename = item['target']
    prompt = item['prompt']
    gt_key = item['ground_truth']

    source = cv2.imread( source_filename)
    target = cv2.imread( target_filename)

        # Do not forget that OpenCV read images in BGR order.
    try:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    except:
        print(source_filename,target_filename)


    # item = dataset[idx]
    # jpg = item['jpg']
    # txt = item['txt']
    # hint = item['hint']
    # print(idx)
# print(txt)
# print(jpg.shape)
# print(hint.shape)
# 1098477
# malinois
# (224, 224, 3)
# (224, 224, 3)


# for idx in range(len(dataset)):
#     item = dataset[idx]
#     source_filename = item['source']
#     target_filename = item['target']
#     prompt = item['prompt']
#     gt_key = item['ground_truth']

#     source = cv2.imread( source_filename)
#     target = cv2.imread( target_filename)

#     # Do not forget that OpenCV read images in BGR order.
#     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

#     # Normalize source images to [0, 1].
#     source = source.astype(np.float32) / 255.0

#     # Normalize target images to [-1, 1].
#     target = (target.astype(np.float32) / 127.5) - 1.0

#     source = resize_image(source, 512)
#     target = resize_image(target, 512)
#     print(idx,source.shape)