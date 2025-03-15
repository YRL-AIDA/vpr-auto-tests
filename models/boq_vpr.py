from models.vpr_interface import VPRModel
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from models import helper
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

from utils.validation import get_validation_recalls
from dataloaders.val.EssexDataset import EssexDataset


class BOQMain(pl.LightningModule):
        def __init__(self,
                    #---- Backbone
                    backbone_arch='dino', 
                    normalize = True,
                    agg_arch='boq',
                    agg_config={
                        'in_channels': 768,  # make sure the backbone has out_channels attribute
                        'proj_channels': 384,
                        'num_queries': 64,
                        'num_layers': 2,
                        'row_dim': 12288//384,
                    },
                    pretrained=True,
                    layers_to_freeze=1,
                    layers_to_crop=[],
                    
                     ):
            super().__init__()
            
            self.encoder_arch = backbone_arch
            self.pretrained = pretrained
            self.layers_to_freeze = layers_to_freeze
            self.layers_to_crop = layers_to_crop
            self.agg_arch = agg_arch
            self.agg_config = agg_config
            
            self.save_hyperparameters() # write hyperparams into a file
    
            
            # ----------------------------------
            # get the backbone and the aggregator
            self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, normalize)
            self.aggregator = helper.get_aggregator(agg_arch, agg_config)
            
        # the forward pass of the lightning model
        def forward(self, x):
            x = self.backbone(x)
            x = self.aggregator(x)
            return x


class BOQMainVPR(VPRModel):
    
    def __init__(self,
                #---- Backbone
                backbone_arch='dino', 
                    normalize = True,
                    agg_arch='boq',
                    agg_config={
                        'in_channels': 768,  # make sure the backbone has out_channels attribute
                        'proj_channels': 384,
                        'num_queries': 64,
                        'num_layers': 2,
                        'row_dim': 12288//384,
                    },
                    pretrained=True,
                    layers_to_freeze=1,
                    layers_to_crop=[],
                
                
                 ):
        self.device = 'cpu'
        self.image_interpolation = T.InterpolationMode.BICUBIC
        self.input_transform_mean = [0.485, 0.456, 0.406] 
        self.input_transform_std = [0.229, 0.224, 0.225]
        if 'resnet' in backbone_arch:
            self.image_size = (320, 320)
        elif 'dino' in backbone_arch:
            self.image_size = (322, 322)
        else:
            self.image_size = (320, 320)

        self.model =  BOQMain(backbone_arch = backbone_arch,pretrained = pretrained,
                          layers_to_freeze = layers_to_freeze, layers_to_crop = layers_to_crop,
                          normalize = normalize, agg_arch = agg_arch, agg_config = agg_config)  
    def input_transform(self):
        return T.Compose([
             T.Resize(self.image_size, interpolation=self.image_interpolation),
    		#T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            
            T.ToTensor(),
            T.Normalize(mean=self.input_transform_mean, std=self.input_transform_std)
        ])
    def get_descriptors(self,dataloader: DataLoader) -> torch.Tensor:
        descriptors = []
        with torch.no_grad():
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output= self.model(imgs.to(self.device))[0].cpu()
                print(output.shape)
                descriptors.append(output)
    
        return torch.cat(descriptors)
        
    def get_embeddings(self, imgs: List[Image]):
        descriptors = []
        with torch.no_grad():
            for img in tqdm(imgs, 'Calculating descritptors...'):
                descriptors.append(self.get_embedding(img))
    
        return descriptors
    def get_embedding(self, img: Image) -> List[float]:
        img = self.input_transform()(img)[np.newaxis,:]
        with torch.no_grad():
            return self.model(img.to(self.device))[0].cpu().tolist()[0]




if __name__ == '__main__':
    mm = BOQMainVPR(backbone_arch='dino', 
            normalize = True,
            agg_arch='boq',
            agg_config={
                'in_channels': 768,  # make sure the backbone has out_channels attribute
                'proj_channels': 384,
                'num_queries': 64,
                'num_layers': 2,
                'row_dim': 12288//384,
            })
    mm.load_model_state_dict('/media/sunveil/Data/header_detection/poddubnyy/postgraduate/VPR/Bag-of-Queries/dinov2_12288.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    mm.set_model_device(device)
    
    def get_val_dataset(dataset_name, input_transform=mm.input_transform()):
        dataset_name = dataset_name.lower()
        
        if 'cross' in dataset_name:
            ds = CrossSeasonDataset(input_transform = input_transform)
        
        elif 'essex' in dataset_name:
            ds = EssexDataset(input_transform = input_transform)
        
        elif 'inria' in dataset_name:    
            ds = InriaDataset(input_transform = input_transform)
        
        elif 'nordland' in dataset_name:    
            ds = NordlandDataset(input_transform = input_transform)
        
        elif 'sped' in dataset_name:
            ds = SPEDDataset(input_transform = input_transform)
        
        elif 'msls' in dataset_name:
            ds = MSLS(input_transform = input_transform)
    
        elif 'pitts' in dataset_name:
            ds = PittsburghDataset(which_ds=dataset_name, input_transform = input_transform)
        else:
            raise ValueError
        
        num_references = ds.num_references
        num_queries = ds.num_queries
        ground_truth = ds.ground_truth
        return ds, num_references, num_queries, ground_truth
    val_dataset_name = 'essex'
    #val_dataset_name = 'nordland'
    batch_size = 10
    
    val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_dataset_name)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)
    
    descriptors = mm.get_descriptors(val_loader)
    print(f'Descriptor dimension {descriptors.shape[1]}')
    
    # now we split into references and queries
    r_list = descriptors[ : num_references].cpu()
    q_list = descriptors[num_references : ].cpu()
    recalls_dict, preds = get_validation_recalls(r_list=r_list,
                                        q_list=q_list,
                                        k_values=[1, 5, 10],
                                        gt=ground_truth,
                                        print_results=True,
                                        dataset_name=val_dataset_name,
                                        )