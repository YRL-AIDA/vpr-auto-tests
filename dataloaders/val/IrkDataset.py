
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os




DATASET_ROOT = f'{os.getcwd()}/datasets/Irk/'
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to IrkDataSet dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')
def to_int_(x):
    try:
        return int(os.path.basename(x).split('.')[0])
    except ValueError as e:
            print (e)
            print (x,os.path.basename(x).split('.'))
class IrkDataset(Dataset):
    def __init__(self, input_transform = None):
        r_dir = 'ref/'
        q_dir = 'query/'

        self.input_transform = input_transform
     
        # reference images names
        self.dbImages = np.array(sorted([r_dir+file for file in os.listdir(DATASET_ROOT+r_dir) if not os.path.isdir(DATASET_ROOT+r_dir+file)],
                                        key=lambda x: to_int_(x)))
        # query images names
        self.qImages = np.array(sorted([q_dir+file for file in os.listdir(DATASET_ROOT+q_dir) if not os.path.isdir(DATASET_ROOT+q_dir+file)],
                                        key=lambda x: to_int_(x)))
        
    
        _,gt = list(zip(*np.load(DATASET_ROOT+'ground_truth.npy', allow_pickle=True)))
        # ground truth
        self.ground_truth = np.array(gt,dtype=object)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)


    def __getitem__(self, index):
        try:
            img = Image.open(DATASET_ROOT+self.images[index])
            img = img.convert('RGB')
            if self.input_transform:
                img = self.input_transform(img)
    
            return img, index
        except Exception as e:
            print(e)
            print(DATASET_ROOT+self.images[index])

    def __len__(self):
        return len(self.images)