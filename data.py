import os
import json
import pandas as pd

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.csv_path = os.path.join(self.data_dir, mode + '.csv')
        self.img_dir = os.path.join(self.data_dir, 'imgs')

        self.data = read_csv(self.csv_path)

        for i in range(len(self.data)):
            self.data[1][i] = os.path.join(self.img_dir, self.data[1][i])


        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        cls, img_path = self.data[0][idx], self.data[1][idx]


        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), cls

