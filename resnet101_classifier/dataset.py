import os
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import pandas as pd
from colorama import Fore
import config
from tqdm import tqdm


label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
class ClassifierDataset(Dataset):
    """
    Custom dataset for handling hyperspectral images.

    Args:
        image_before_dir (str): Directory containing 'before' images.
        image_after_dir (str): Directory containing 'after' images.
        transform (bool): Flag for data augmentation.

    Attributes:
        image_before_dir (str): Directory containing 'before' images.
        image_after_dir (str): Directory containing 'after' images.
        transform (bool): Flag for data augmentation.
        images_before (list): List of 'before' image filenames.
        images_after (list): List of 'after' image filenames.
    """

    def __init__(self, image_dir, channels: int, c_step: int, transform=False, state='before'):
        self.image_dir = image_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        self.images = [file for file in Path(image_dir).glob(f'*/{state}.npy')]
        self.images.remove(Path('/home/ARO.local/tahor/PycharmProjects/data/pair_data/box_19_class_B_num_88/before.npy'))
        print('len images: ', len(self.images))
        self.channels = channels
        self.c_step = c_step
        ###
        self.images = self.images[10:20]
        ###
        # self.images = list(filter(lambda x: np.load(str(x)).ndim == 3, self.images))

    def load_all_images(self, images):
        valid = 0
        not_valid = 0
        for image_path in tqdm(images, desc='Classifier load images'):
            im = np.load(image_path)
            if im.ndim == 3:
                self.images.append(im)
                valid += 1
            else:
                not_valid += 1
        print(f'({valid}/{not_valid + valid}) valid images')


    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get a sample from the dataset at a given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing 'before' and 'after' images as numpy arrays.
        """
        image_path = self.images[index]

        image = config.initial_transform(image=np.load(str(image_path)
                                                       )[:, :, [i for i in range(self.channels) if i % self.c_step == 0]] / 4096)['image']

        label = label_dict[str(image_path).split('/')[-2].split('_')[3]]

        if self.transform:
            image = config.transform_only_input(image=image)["image"]

        return image.transpose((2, 1, 0)).astype(np.float32), label




