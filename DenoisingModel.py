import os
import sys
import torch
import numpy as np

# Path initialization
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT_DIR)
from model import EncoderDecoderModel


class DenoisingModel():

    def __init__(self, device):
        r""" Initialization of Denoising Model.
        """

        self.model = EncoderDecoderModel()
        self.model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "weights.pth"), map_location=device))

        self.model = self.model.to(device)
        self.model.eval()
    
    def predict(self, data_path, device):
        r""" Loads the spectrogram from data_path, preprocesses the spectrogram and
        applies NN model to obtain denoised sample.
        
        Inputs:
            data_path: str
                Path to the data.
            
            device: torch.device
                Device for calculation.
        
        Outputs:
            signal_pred: numpy.ndarray
                Predicted spectrogram.
            
            mae: float
                Sum of Absolute Error between input and predicted signal (is needed for classification).
            
        """
        
        signal = self.get_file(data_path)
        x = self.preprocessing(signal)
        
        with torch.no_grad():
            signal_pred = self.model(x.to(device))
        
        signal_pred = signal_pred[0, 0, :, :]
        signal_pred = signal_pred.cpu().numpy()

        mae = abs(signal_pred - signal).sum()
        return signal_pred, mae
            
    @staticmethod
    def get_file(path):
        r""" Loads .npy file and adds zeros if it is too small to be fed into the network.
        """
        
        signal = np.load(path)
        
        # Check the size
        if signal.shape[0] < 60:
            signal_expanded = np.zeros((60, signal.shape[1]))
            signal_expanded[:signal.shape[0], :] = signal
            return signal_expanded
        else:
            return signal
    
    @staticmethod
    def preprocessing(x):
        r""" Transfer array into tensor.
        """
        
        x = x[None, None].astype(np.float32)
        x = torch.tensor(x)

        return x
        