from .Dataset_Parton_Level import Dataset_PartonLevel
from .Dataset_Reco_Level import Dataset_RecoLevel
import torch
from torch.utils.data import Dataset, DataLoader

class DatasetCombined(Dataset):
    def __init__(self, root, dev=None, debug=False, dtype=None):

        self.reco_data = Dataset_RecoLevel(root, dev=dev, debug=debug, dtype=dtype)
        self.parton_data = Dataset_PartonLevel(root, dev=dev, debug=debug, dtype=dtype)
        
    def __getitem__(self, index):
        return *self.parton_data[index], *self.reco_data[index]

    def __len__(self):
        return len(self.parton_data)