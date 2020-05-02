import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioDATA(Dataset):
    
    def __init__(self, data_dir, fixed_size=1000):
        
        super().__init__()
        
        self.data_dir = data_dir

        self.data_paths_noisy = self.get_data_paths(os.path.join(data_dir, "noisy"))
        self.data_paths_clean = self.get_data_paths(os.path.join(data_dir, "clean"))
        
        self.fixed_size = fixed_size
        
    def __len__(self):
        
        return len(self.data_paths_clean)
    
    def expand_to_fixed_size(self, x):
        r""" Cats the input spectrogram to the fixed size. 
        """
        
        if x.shape[0] > self.fixed_size:
            return x[:self.fixed_size, :]
        else:
            x_expanded = np.zeros((self.fixed_size, x.shape[1]))
            x_expanded[:x.shape[0], :] = x
            return x_expanded
    
    def __getitem__(self, index):
        
        data_path_clean = self.data_paths_clean[index]
        signal_clean = np.load(data_path_clean)
        signal_clean = self.expand_to_fixed_size(signal_clean).astype(np.float32)
        
        if index % 5 == 0:
            signal_noisy = np.copy(signal_clean)
            
        else:
            data_path_noisy = self.data_paths_noisy[index]
            signal_noisy = np.load(data_path_noisy)
            signal_noisy = self.expand_to_fixed_size(signal_noisy).astype(np.float32)

        return signal_noisy[None], signal_clean[None]
    
    @staticmethod
    def get_data_paths(dir_path):
        r""" Get all data paths to .npy files in the directory.
        """
        
        data_paths_list = []
        
        for folder in os.listdir(dir_path):
            folder_path = os.path.join(dir_path, folder)
            
            for file_neme in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_neme)
                if file_neme.split(".")[-1] == "npy":
                    data_paths_list.append(file_path)

        return data_paths_list
    
    
    