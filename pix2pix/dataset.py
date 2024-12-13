import os
from typing import Union

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import pandas as pd
from colorama import Fore
from torchvision.models.shufflenetv2 import channel_shuffle

import config as config
# from paths.persimmon_paths import hpc_paths_for_data_after, hpc_paths_for_data_before
from sklearn.model_selection import train_test_split

# before_path = hpc_paths_for_data_before['single_fruit_path']
# after_path = hpc_paths_for_data_after['single_fruit_path']



class Pix2PixDataset(Dataset):
    def __init__(self, channels: int | list , c_step=1 ,image_dir="/home/ARO.local/tahor/PycharmProjects/data/pair_data", transform=None):
        self.channels = channels
        self.c_step = c_step
        self.before = []
        self.after = []  # file for file in Path(image_dir).glob("*") if (len(list(Path(file).glob("*.npy"))) == 2)]
        for before_path in Path(image_dir).glob('**/before.npy'):
            if before_path.with_name('after.npy').exists() and 1.2 >= before_path.with_name('after.npy').stat().st_size / before_path.stat().st_size >= 0.8:
                self.before.append(before_path)
                self.after.append(before_path.with_name('after.npy'))
            # if len(self.before) == 5 and len(self.after) == 5:
            #     break
        self.transform = transform

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.before)

    def __getitem__(self, index):
        # p = nn.AvgPool3d((10, 1, 1), stride=(10, 1, 1))
        if self.channels is not list:
            before = np.load(str(self.before[index]))[:, :, [i for i in range(self.channels) if i % self.c_step == 0]] / 4096
            after = np.load(str(self.after[index]))[:, :, [i for i in range(self.channels) if i % self.c_step == 0]] / 4096
        else:
            before = np.load(str(self.before[index]))[:, :, self.channels]
            after = np.load(str(self.after[index]))[:, :, self.channels]

        before = config.initial_transform(image=before)['image']
        after = config.initial_transform(image=after)['image']
        if before.shape != after.shape:
            print()
        if self.transform is not None:
            augmentations = config.both_transforms(image=before, image0=after)

            return (augmentations["image"].transpose((2, 1, 0)).astype(np.float32),
                     augmentations["image0"].transpose((2, 1, 0)).astype(np.float32))
        else:
            return (before.transpose((2, 1, 0)).astype(np.float32),
                    after.transpose((2, 1, 0)).astype(np.float32))



test_dir = "/home/ARO.local/tahor/PycharmProjects/data/pair_data"
#
#
# dataset = Pix2PixDataset(test_dir, transform=False)
# loader = DataLoader(dataset, batch_size=2)
# print('dataset [0][1]', type(dataset[0][1]))
# print('dataset [0][0]', dataset[0][0].shape)
# print('dataset [10][0]', dataset[10][0].shape)
# print('dataset [10][1]', dataset[10][1].shape)
# print(loader.batch_size)

# def train_val_dataset(dataset, split:float) ->dict:
#     train_idx, val_idx = train_test_split(range(len(dataset.before)), test_size=split)
#     datasets = {}
#     datasets['train'] = Subset(dataset, train_idx)
#     datasets['val'] = Subset(dataset, val_idx)
#     return datasets




# set = Subset(dataset=dataset, indices=[i for i in range(len(dataset.before)) if i%15 == 0])
# c = 0
# for i in set:
#     # print(i[0].shape)
#     c += 1
#     # plt.imshow(i[0][40, :, :]), plt.show()
#     print("c: ", c)
# for x, y in loader:
#     print('x: ', x[1][0].shape)
#     print('y:', y.shape)
