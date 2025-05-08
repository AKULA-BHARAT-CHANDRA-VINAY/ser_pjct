import os
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa

class myDataset(Dataset):
    def __init__(self, path, batch_size, uttr_len=300, fre_size=200):
        self.batch_size = batch_size
        self.path = path
        self.file_list = sorted(os.listdir(path))
        self.uttr_len = uttr_len
        self.fre_size = fre_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.file_list[idx])
        data = np.load(file_path)
        data = data[:, :self.fre_size]  # Trim frequency dimension
        label = int(self.file_list[idx].split('_')[0])  # Label from filename
        data = torch.tensor(data, dtype=torch.float).transpose(0, 1)
        data = self._pad_sequence(data)
        return data, label

    def _pad_sequence(self, x):
        length = x.shape[1]
        if length >= self.uttr_len:
            x = x[:, :self.uttr_len]
        else:
            pad = torch.zeros(x.shape[0], self.uttr_len - length)
            x = torch.cat([x, pad], dim=1)
        return x