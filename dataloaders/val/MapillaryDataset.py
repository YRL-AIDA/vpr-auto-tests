from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

import os

CITY = "moscow"  # msls

GT_ROOT = f'{os.getcwd()}/datasets/msls_val/'

DATASET_ROOT = GT_ROOT


class MSLS(Dataset):
    def __init__(self, input_transform = None):
        
        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT + f'{CITY}_dbImages.npy')
        # self.dbImages = np.load(GT_ROOT + f'modified_{CITY}_dbImages.npy')
        self.qIdx = np.load(GT_ROOT + f'{CITY}_qIdx.npy')
        self.qImages = np.load(GT_ROOT + f'{CITY}_qImages.npy')
        # self.ground_truth = np.load(GT_ROOT + f'{CITY}_pIdx.npy', allow_pickle=True)
        self.ground_truth = np.load(GT_ROOT + f'modified_{CITY}_pIdx.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)