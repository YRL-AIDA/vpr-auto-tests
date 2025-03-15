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

class VLADBuffMain(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """
    
    def __init__(
        self,
        # ---- Backbone
        backbone_arch='segvlad', 
        agg_arch='salad',
        backbone_config={},
        agg_config={
            'num_channels': 768,  # make sure the backbone has out_channels attribute
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256
        },
        args=None,
    ):
        super().__init__()
        self.IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.save_hyperparameters()  # write hyperparams into a file
        
        self.args = args
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)

        if "netvlad" in agg_arch.lower():
            # Create an instance of the AggConfig class
            self.aggLayer = helper.get_aggregator(agg_arch, agg_config)

            # cluster using gsv single city
            if agg_config.initialize_clusters:
                from dataloaders.GSVCitiesDataset import GSVCitiesDataset
                # Instantiate GSVCitiesDataset with the desired city
                selected_city = "London"  # Replace with the city you want
                single_city_dataset = GSVCitiesDataset(
                    cities=[selected_city],
                    img_per_place=1,  # Adjust as needed
                    min_img_per_place=1,  # Adjust as needed
                    random_sample_from_each_place=True,
                    transform=T.Compose(
                        [
                            T.Resize(
                                self.args.resize,
                                interpolation=T.InterpolationMode.BILINEAR,
                            ),
                            T.ToTensor(),
                            T.Normalize(
                                mean=self.IMAGENET_MEAN_STD["mean"],
                                std=self.IMAGENET_MEAN_STD["std"],
                            ),  # Adjust mean and std if needed
                        ]
                    ),
                )
                self.aggLayer.initialize_netvlad_layer(
                    agg_config, single_city_dataset, self.backbone
                )

            if agg_config.l2 == "before_pool":
                self.aggLayer = nn.Sequential(L2Norm(), self.aggLayer, Flatten())
            elif agg_config.l2 == "after_pool":
                self.aggLayer = nn.Sequential(self.aggLayer, L2Norm(), Flatten())
            elif agg_config.l2 == "onlyFlatten":
                self.aggLayer = nn.Sequential(self.aggLayer, Flatten())

            if agg_config.useFC:  # fc_dim used so have a NV agg layer in nn.Seq aggregation layer
                if agg_config.nv_pca is not None:
                    netvlad_output_dim = agg_config.nv_pca

                netvlad_output_dim *= agg_config.clusters_num

                if agg_config.fc_output_dim == 0:
                    fcLayer = nn.Identity()
                    agg_config.fc_output_dim = netvlad_output_dim
                else:
                    fcLayer = nn.Linear(netvlad_output_dim, agg_config.fc_output_dim)

                self.aggregator = nn.Sequential(self.aggLayer, fcLayer, L2Norm())
            else:  # no fc_dim used
                self.aggregator = self.aggLayer
                del self.aggLayer

                # check wpca layer to be used? / can be used during evaluation only
                if agg_config.wpca:
                    if args.nv_pca is not None:
                        netvlad_output_dim = args.nv_pca
                    else:
                        netvlad_output_dim = agg_config.dim
                    netvlad_output_dim = agg_config.clusters_num * netvlad_output_dim
                    pca_conv = nn.Conv2d(
                        netvlad_output_dim,
                        agg_config.num_pcs,
                        kernel_size=(1, 1),
                        stride=1,
                        padding=0,
                    )
                    self.WPCA = nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
        else:  # SALAD
            self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class VLADBuffMainVPR(VPRModel):
    def __init__(self,
                #---- Backbone
                backbone_arch='segvlad', 
                agg_arch='salad',
                agg_config={
                    'num_channels': 768,  # make sure the backbone has out_channels attribute
                    'num_clusters': 64,
                    'cluster_dim': 128,
                    'token_dim': 256,
                },
                args=None):
        self.device = 'cpu'
        self.image_interpolation = T.InterpolationMode.BILINEAR
        self.input_transform_mean = [0.485, 0.456, 0.406] 
        self.input_transform_std = [0.229, 0.224, 0.225]
        self.image_size = (320, 320)

        self.model =  VLADBuffMain(backbone_arch = backbone_arch,
                        agg_arch = agg_arch, agg_config = agg_config,args=args)  
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
    mm = VLADBuffMainVPR(backbone_arch='segvlad', 
                agg_arch='salad',
                agg_config={
                    'num_channels': 768,  # make sure the backbone has out_channels attribute
                    'num_clusters': 64,
                    'cluster_dim': 128,
                    'token_dim': 256,
                })
    mm.load_model_state_dict('/media/sunveil/Data/header_detection/poddubnyy/postgraduate/VPR/segvlad/dino_salad.ckpt')
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