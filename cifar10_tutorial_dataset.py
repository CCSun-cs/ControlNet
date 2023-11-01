import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3


class MyDataset(Dataset):
    def __init__(self):
        self.data = []

        # with open('/egr/research-optml/sunchan5/AEG/dataset/cifar10_final/train_controlnet/prompt_cifar10_lable_original_label_target.json', 'rt') as f:
            #prompt_cifar10_noprompt
        with open('/egr/research-optml/sunchan5/AEG/dataset/cifar10_final/train_controlnet/prompt_cifar10_noprompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        gt_key = item['ground_truth']

        source = cv2.imread( source_filename)
        target = cv2.imread( target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        source = resize_image(source, 512)
        target = resize_image(target, 512)

        # return dict(jpg=target, txt=prompt, hint=source)
        return dict(jpg=target, txt=prompt, hint=source,gt_txt=gt_key)


