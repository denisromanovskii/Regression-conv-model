import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

class DatasetReg(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_files = os.listdir(self.path)
        if 'coords.json' in self.list_files:
            self.list_files.remove('coords.json')

        self.len_dataset = len(self.list_files)
        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        name_file = self.list_files[index]
        path_img = os.path.join(self.path, name_file)

        img = Image.open(path_img)
        coord = self.dict_coords[name_file]
        if self.transform is not None:
            img = self.transform(img)
            coord = torch.tensor(coord, dtype=torch.float32)
        return img, coord

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, ), std=(0.5, ))
])

dataset = DatasetReg('dataset', transform=transform)

train_data, val_data, test_data = random_split(dataset, [0.7, 0.1, 0.2])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, pin_memory=True)