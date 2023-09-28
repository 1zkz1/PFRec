from torch.utils.data import Dataset
import torch
import numpy as np

class Amazon_MyModel_GAN8(Dataset):

    def __init__(self, rating: np.numarray):
        super(Amazon_MyModel_GAN8, self).__init__()
        self.len = rating.shape[0]
        self.data = torch.tensor(rating)

    def __getitem__(self, idx):
        return self.data[idx], idx

    def __len__(self):
        return self.len

