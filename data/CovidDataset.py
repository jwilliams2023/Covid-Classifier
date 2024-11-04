import torch
import pandas as pd
from torch.utils.data import Dataset


class CovidDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

        self.features = self.data.drop('DIED', axis=1).values
        self.labels = self.data['DIED'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y
