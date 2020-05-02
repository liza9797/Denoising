import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
import torch
from torch import nn
from collections import defaultdict
from tqdm import tqdm 

def init_dir(model, weight_folder):

    # Init
    arch_name = model.__module__
    model_full_name = "{}_{}".format(arch_name, datetime.now().strftime('%d.%m.%Y.%H:%M'))
    
    model_dir = os.path.join(weight_folder, model_full_name)
    
    # Make dirs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs))) 


def update_df(df, metrics, epoch, lr, phase, epoch_samples, time, path):
    df.loc[df.shape[0]] = [epoch, lr, phase, 
                           metrics['loss'] / epoch_samples, time]
    df.to_csv(path, index=False)

def train_model(model, dataloaders, optimizer, scheduler, 
                 num_epochs,
                 print_per_iter=1, 
                 device='cpu', 
                 phase_range=['train', 'val'], 
                 phase_to_save_model=['val'],
                 model_dir=None, max_saved_models=3):
    

    # Initial best loss
    best_losses, num_saved_models = {}, {}
    for phase in phase_to_save_model:
        best_losses[phase] = [1000000.]
        num_saved_models[phase] = 0
    
    path_to_save_df = os.path.join(model_dir, "train_history.csv")
    
    # Create dataframe
    df = pd.DataFrame(columns=["epoch", "lr", "phase", 'loss', "time"])
    criterion = nn.MSELoss(reduction="sum")
    
    # Start trainig loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since_ep = time.time()

        for phase in phase_range:
            since = time.time()
            
            if 'train' in phase:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    print("LR", param_group['lr'])
                model.train() 
            
            else:
                model.eval()   

            metrics = defaultdict(float)
            epoch_samples = 0

            for i, (X, Y) in tqdm(enumerate(dataloaders[phase])):

                Y = Y.to(device)
                X = X.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled('train' in phase):
                    
                    outputs = model(X)
                 
                    # Update metrics
                    loss = criterion(Y.float(), outputs)
                    metrics['loss'] += float(loss)
                    epoch_samples += 1
                    
                    if 'train' in phase:
                        loss.backward(retain_graph=True)
                        optimizer.step()
                
                # Print and save metrics
                if i % print_per_iter == 0:
                    print_metrics(metrics, epoch_samples, phase)
                    time_elapsed = time.time() - since
                    update_df(df, metrics, epoch, lr, phase, epoch_samples, time=None,
                              path=path_to_save_df)
                    
                del X, Y, outputs, loss
            
            # End for phase
            time_elapsed = time.time() - since
            print('{}: {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # Save data
            update_df(df, metrics, epoch, lr, phase, epoch_samples, 
                      path=path_to_save_df, time=time_elapsed)
            
            # Save best model
            if phase in phase_to_save_model and (epoch_loss < np.array(best_losses[phase])).any():
                if 1000000. in best_losses[phase]:
                    best_losses[phase].remove(1000000.)

                # Save new best model
                print("--saving best model--")
                best_losses[phase].append( epoch_loss )
                model_suffix = "weight.epoch_{}_loss_{}_{}.pth".format(epoch+1, phase, epoch_loss)
                model_path = os.path.join(model_dir, model_suffix)

                torch.save(model.state_dict(), model_path)
                num_saved_models[phase] += 1

                # If there are more then max_saved_models saved models, remove the worst one
                if num_saved_models[phase] > max_saved_models:
                    max_loss = 0.0
                    for file_name in os.listdir(model_dir):
                        if "loss" in file_name and phase in file_name:
                            loss = file_name.split("_")[-1][:-4]
                            loss = float(loss)
                            if loss > max_loss:
                                max_loss = loss
                                model_with_max_loss = os.path.join(model_dir, file_name)
                    os.remove(model_with_max_loss)
                    best_losses[phase].remove(max_loss)
                    num_saved_models[phase] -= 1

        time_elapsed = time.time() - since_ep
        print('Epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        
    print('Best val loss: {:4f}'.format( np.min(best_losses["val"]) ))
    