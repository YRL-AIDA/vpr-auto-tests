from dataclasses import dataclass,field,asdict
import json
from typing import List, Dict
import models
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from utils.validation import get_validation_recalls
from models import helper
from utils import utils
import models
import argparse


if __name__ == "__main__":

    # Создаем парсер аргументов

    parser = argparse.ArgumentParser(description="Простой скрипт для демонстрации argparse.")

    parser.add_argument("conf_path", type=str, help="Путь к файлу с настройками тестирования")


    # Парсим аргументы

    args = parser.parse_args()


    # Вызываем основную функцию с аргументом

    print(args.conf_path)
    conf = utils.load_from_json(args.conf_path)
    for test_name in conf.tests:
        for model_conf in conf.models:
            model_conf = utils.ModelItem(**model_conf)
            device = torch.device(model_conf.device)
            print(f'------- {model_conf.title} -------')
            print(device)
            model = models.MODELS_PULL[model_conf.model_name](**model_conf.model_conf)
            model.load_model_state_dict(model_conf.model_weight_path)
            model.set_model_device(device)
            for dataset_name in conf.datasets:
                utils.run_test(model,dataset_name,test_name)
