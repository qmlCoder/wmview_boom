import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# class SmilesDataset(Dataset):
#     def __init__(self, path):
#         super().__init__()
#         self.dataset = torch.load(path)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

class SmilesDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
# dataset = SmilesDataset(path = '/export/home/zhengym/model/fusion/FUSED-mpnn/data/data_pt/new_123679_49.pt')             # 旧数据
dataset = SmilesDataset(path = '/export/home/zhengym/model/fusion/GRMP/data/new_data/pt/123679_49_new.pt')          # 新数据

data_size = len(dataset)

def get_idx_split(data_size, train_size, valid_size, seed):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(data_size, generator=generator)
    
    train_idx = ids[:train_size]
    val_idx = ids[train_size:train_size + valid_size]
    test_idx = ids[train_size + valid_size:]
    
    split_dict = {
        'train': train_idx.long(),
        'valid': val_idx.long(),
        'test': test_idx.long()
    }
    return split_dict

split_idx = get_idx_split(data_size, train_size=98945, valid_size=12367, seed=641)

train_dataset = torch.utils.data.Subset(dataset, split_idx['train'].tolist())
valid_dataset = torch.utils.data.Subset(dataset, split_idx['valid'].tolist())
test_dataset = torch.utils.data.Subset(dataset, split_idx['test'].tolist())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
