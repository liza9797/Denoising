import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

# Path initialization
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT_DIR)
from dataloader import AudioDATA
from model import EncoderDecoderModel
from functional import init_dir, train_model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--device', type=str, default="cpu", action='store',
                        help='Device to use.')
    
    parser.add_argument('-l', '--path-to-dataset', type=str, default="/dataset", action='store',
                        help='Path to dataset.')
    parser.add_argument('-s', '--path-to-results', type=str, default="/results", action='store',
                        help='Path to save the weights and training history.')
    parser.add_argument('--model-weights', type=str, default=None, action='store',
                        help='Path to the model weights to load into the model. If it is not specified, model will be trained from scratch.')
    
    parser.add_argument('-b', '--batch-size', type=int, default=16, action='store',
                        help='Batch size.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5, action='store',
                        help='Initial learning rate.')
    parser.add_argument('--scheduler-step', type=int, default=30, action='store',
                        help='Sheduler step, shows once in how many epoches learning rate will be multiplied by 0.1')

    parser.add_argument('-e', '--epochs', type=int, default=100, action='store',
                        help='Number of epochs.')
    
    


    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    # Device
    device = torch.device(args.device)
    
    # Data
    phase_range = ["train", "val"]
    datasets = {phase: AudioDATA(data_dir=args.path_to_dataset + phase) 
                    for phase in phase_range}
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=True) 
                    for phase in phase_range}
    print(len(datasets["train"]))
    
    # Model
    model = EncoderDecoderModel()
    if args.model_weights:
        model.load_state_dict(torch.load(args.model_weights))
    
    # Init dir to the model training session
    model_dir = init_dir(model, weight_folder=args.path_to_results)
    
    # Train parameters
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    num_epochs = args.epochs
    
    # Train
    train_model(model.to(device), dataloaders, optimizer, scheduler, 
                num_epochs,
                print_per_iter=2, 
                device=device, 
                phase_range=['train', 'val'], 
                phase_to_save_model=['train', 'val'],
                model_dir=model_dir, 
                max_saved_models=3)