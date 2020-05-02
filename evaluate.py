# import librosa # for mel-spectrogram estimation
# import soundfile

import numpy as np
import pandas as pd
import os
import sys
import torch
from tqdm import tqdm

# Path initialization
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT_DIR)
from DenoisingModel import DenoisingModel
        
        
def get_data_paths(current_path, data_paths_list):
    r""" Gets all pathes of .npy files from the directory.
    """

    if os.path.isdir(current_path):
        
        for file_name in os.listdir(current_path):
            path = os.path.join(current_path, file_name)
            get_data_paths(path, data_paths_list)
    else:
        if current_path.split(".")[-1] == "npy":
            data_paths_list.append(current_path)
    
    
def evaluation(model, path_to_dataset, path_to_results, device):
    r""" Evaluate model on the data from path_to_datase directory,
    save results to path_to_results directory.
    """
    
    # Create  path_to_results/denoised directory if it does not exist
    path_to_results_denoised = os.path.join(path_to_results, "denoised")
    if not os.path.exists(path_to_results_denoised):
        os.makedirs(path_to_results_denoised)
    
    # Create file with results
    df = pd.DataFrame(columns=["file_name", "denoised_file", "mae"])
    
    # Get all pathes to files of the dataset
    data_paths_list = [] 
    get_data_paths(path_to_dataset, data_paths_list)
    
    # Calculation
    for i, data_path in tqdm(enumerate(data_paths_list)):
        
        prediction, mae = model.predict(data_path, device)
        path_to_results_file = os.path.join(path_to_results_denoised, "file{}_denoised.npy".format(i))
        
        # Save data for denoised file
        np.save(path_to_results_file, prediction)
        df.loc[df.shape[0]] = [data_path, path_to_results_file, mae]
        
    # Save results
    path_to_results_df = os.path.join(path_to_results, "results.csv")
    df.to_csv(path_to_results_df, index=False)
        

if __name__ == "__main__":
    
    path_to_dataset = "../data/val/noisy/"
    path_to_results = "results/"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenoisingModel(device)
    
    # Eval
    evaluation(model, 
               path_to_dataset=path_to_dataset, 
               path_to_results=path_to_results, 
               device=device)

    
    
    
    
    
    