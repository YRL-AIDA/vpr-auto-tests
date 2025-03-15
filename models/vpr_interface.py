from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import List
import torchvision.transforms as T
import torch
from PIL import Image


class VPRModel(ABC):
    ''' Base interface to use models in auto-test framework '''
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        
    @abstractmethod
    def input_transform(self) -> T.Compose:
        ''' Input image transform method'''
        pass

    @abstractmethod
    def get_embedding(self,img: Image) -> List[float]:
        ''' Method to geting embedding of a single image'''
        pass

    @abstractmethod
    def get_embeddings(self,imgs:List[Image]) -> List[List[float]]:
        ''' Method to geting embeddings of image list'''
        pass
        
    @abstractmethod
    def get_descriptors(self,val_loader: DataLoader) -> torch.Tensor:
        ''' A method for obtaining image embeddings for model testing on datasets'''
        pass
        
    def load_model_state_dict(self,path_to_weight: str):
        state_dict = torch.load(path_to_weight) 

        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def set_model_device(self,device:str):
        self.device = device
        self.model = self.model.to(self.device)