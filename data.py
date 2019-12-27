import os
import pandas as pd

import torchvision.transforms as transforms
from operator import itemgetter
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.csv_path = os.path.join(self.data_dir, mode + '.csv')
        self.img_dir = os.path.join(self.data_dir, 'imgs')

        data_F = read_csv(self.csv_path)

        for i in range(len(data_F)):
            data_F[1][i] = os.path.join(self.img_dir, data_F[1][i])

        data_S = data_F.values.tolist()
        data_S.sort(key=itemgetter(0))
        last_num = -1
        cnt = 0
        self.data = []
        for i in range(len(data_S)):
            if(data_S[i][0] == last_num) & (cnt < 3):
                cnt += 1
            else:
                self.data.append([])
                cnt = 0
            if i == (len(data_S)-1):
                if cnt != 3:
                    for e in range(3-cnt):
                        self.data[-1].append(data_S[i-e-1])
            elif (data_S[i][0] != data_S[i+1][0]) & (cnt != 3):
                for e in range(3-cnt):
                    self.data[-1].append(data_S[i-e-1])
            self.data[-1].append(data_S[i])
            last_num = data_S[i][0]


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
        images = []
        for i in range(len(self.data[idx])):
            im_2 = []
            cls, img_path = self.data[idx][i][0], self.data[idx][i][1]
            img = Image.open(img_path).convert('RGB')
            img_t = self.transform(img)
            images.append(img_t)

        images = torch.stack(images)


        ''' read image '''
        return images, cls

