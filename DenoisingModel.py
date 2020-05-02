import os
import torch
import numpy as np

# Path initialization
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT_DIR)
from model import EncoderDecoderModel


class DenoisingModel():
    def __init__(self, device):
        
        self.model = EncoderDecoderModel()
        self.model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "weights.pth"), map_location=device))

        self.model = self.model.to(device)
        self.model.eval()
    
    def predict(self, data_path, device):
        
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
        
        x = x[None, None].astype(np.float32)
        x = torch.tensor(x)

        return x
        