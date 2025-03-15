import argparse
import json
import os

import torch

from dataset import ECG_DataModule
from trainer import Trainer

def main(dataset="icentia11k", dataset_path="./data/icentia11k.pkl", batch_size=32, seed=None):
    # dataloader
    data_module = ECG_DataModule(dataset, dataset_path, batch_size=batch_size, seed=seed)
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()


if __name__ == "__main__":
    main()
