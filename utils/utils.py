from dataclasses import dataclass,field,asdict
from typing import List, Dict
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from utils.validation import get_validation_recalls
from models import helper
from models import VPRModel
from dataloaders.val.EssexDataset import EssexDataset
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.IrkDataset import IrkDataset
import models
@dataclass
class ModelItem:
    title: str
    model_name: str
    model_weight_path: str
    model_conf: Dict = field(default_factory=dict)
    device: str = field(default="cpu")
    def to_serializable_dict(self):

        # Преобразуем device в строку

        serializable_dict = asdict(self)

        serializable_dict['device'] = str(self.device)  # Преобразуем device в строку

        return serializable_dict
@dataclass
class TestArgs:
    models: List[ModelItem] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)


def load_from_json(filename: str) -> TestArgs:
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return TestArgs(**data)



def get_val_dataset(dataset_name, input_transform):
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
        elif 'irk' in dataset_name:
            ds = IrkDataset(input_transform = input_transform)
        else:
            raise ValueError
        
        num_references = ds.num_references
        num_queries = ds.num_queries
        ground_truth = ds.ground_truth
        return ds, num_references, num_queries, ground_truth

def run_test(model:models.VPRModel,dataset_name:str,test_name:str,batch_size:int = 10) -> dict:
    if test_name == 'recall':
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(dataset_name,model.input_transform())
        val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)
        descriptors = model.get_descriptors(val_loader)
        print(f'Descriptor dimension {descriptors.shape[1]}')
        
        # now we split into references and queries
        r_list = descriptors[ : num_references].cpu()
        q_list = descriptors[num_references : ].cpu()
        recalls_dict, preds = get_validation_recalls(r_list=r_list,
                                            q_list=q_list,
                                            k_values=[1, 5, 10],
                                            gt=ground_truth,
                                            print_results=True,
                                            dataset_name=dataset_name,
                                            )
    return recalls_dict, preds