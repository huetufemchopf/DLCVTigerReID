import os
import pandas as pd
import torchvision.transforms as transforms
from operator import itemgetter
from torch.utils.data import Dataset
from PIL import Image
from data_utils import RandomErasing
import torch
import random

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


def trans(img):
    w, h = img.size
    if h > w:
        diff = int((h-w)/2)
        transform_input = transforms.Compose([
            transforms.Pad((diff, 0), fill=0, padding_mode='constant'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        diff = int((w-h)/2)
        transform_input = transforms.Compose([
            transforms.Pad((0, diff), fill=0, padding_mode='constant'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    return transform_input(img)


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        data_dir = args.data_dir
        csv_path = os.path.join(data_dir, mode + '.csv')
        img_dir = os.path.join(data_dir, 'imgs')

        data_F = read_csv(csv_path)
        self.imgs_names = read_csv(csv_path)[1]
        for i in range(len(data_F)):
            data_F[1][i] = os.path.join(img_dir, data_F[1][i])

        # Group Labels for the training set
        if mode == 'train':
            data_S = data_F.values.tolist()
            data_S.sort(key=itemgetter(0))

            last_num = -1
            idx = -1
            # Change labels to go from 0 to number of different labels
            for i in range(len(data_S)):
                if data_S[i][0] != last_num:
                    idx += 1
                    last_num = data_S[i][0]
                data_S[i][0] = idx

            last_num = -1
            cnt = 0
            num = args.label_group - 1
            self.data = []
            for i in range(len(data_S)):
                if(data_S[i][0] == last_num) & (cnt < num):
                    cnt += 1
                else:
                    self.data.append([])
                    cnt = 0
                if i == (len(data_S)-1):
                    if cnt != num:
                        for e in range(num-cnt):
                            self.data[-1].append(data_S[i-e-1])
                elif (data_S[i][0] != data_S[i+1][0]) & (cnt != num):
                    for e in range(num-cnt):
                        self.data[-1].append(data_S[i-e-1])
                self.data[-1].append(data_S[i])
                last_num = data_S[i][0]
        else:
            self.data = data_F
        img_size = 256
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            RandomErasing
        ])

        self.transform_t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        if self.mode == 'train':
            images = []
            for i in range(len(self.data[idx])):
                cls, img_path = self.data[idx][i][0], self.data[idx][i][1]
                img = Image.open(img_path).convert('RGB')
                img_t = self.transform(img)
                images.append(img_t)

            images = torch.stack(images)
        else:
            img = Image.open(self.data[1][idx]).convert('RGB')
            images = self.transform_t(img)
            cls = self.imgs_names[idx]

        ''' read image '''
        return images, cls


# Dataloader with random geting of images from the same label, slight difference.
class DATA2(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.args = args
        self.mode = mode
        data_dir = args.data_dir
        csv_path = os.path.join(data_dir, mode + '.csv')
        img_dir = os.path.join(data_dir, 'imgs')

        data_F = read_csv(csv_path)
        self.imgs_names = read_csv(csv_path)[1]
        for i in range(len(data_F)):
            data_F[1][i] = os.path.join(img_dir, data_F[1][i])

        # Group Labels for the training set
        if mode == 'train':
            data_S = data_F.values.tolist()
            data_S.sort(key=itemgetter(0))

            # Change labels to go from 0 to number of different labels
            '''
            for i in range(len(data_S)):
                if data_S[i][0] != last_num:
                    idx += 1
                    last_num = data_S[i][0]
                data_S[i][0] = idx
            '''
            last_num = -1
            self.data = []
            for i in range(len(data_S)):
                if (data_S[i][0] != last_num):
                    self.data.append([])
                self.data[-1].append(data_S[i])
                last_num = data_S[i][0]
        else:
            self.data = data_F
        img_size = 224
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            # RandomErasing
        ])

        self.transform_t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        if self.mode == 'train':
            images = []
            random_ids = random.sample(range(0, len(self.data[idx])), self.args.label_group)
            for i in random_ids:
                cls, img_path = self.data[idx][i][0], self.data[idx][i][1]
                img = Image.open(img_path).convert('RGB')
                img_t = self.transform(img)
                images.append(img_t)
            images = torch.stack(images)
        else:
            img = Image.open(self.data[1][idx]).convert('RGB')
            images = self.transform_t(img)
            cls = self.imgs_names[idx]

        ''' read image '''
        return images, cls


#Dataloader with random geting of images from the same label, slight difference.
class DATA_un(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.args = args
        data_dir = args.data_dir
        csv_path = os.path.join(data_dir, 'unsupervised.csv')
        self.img_dir = os.path.join(data_dir, 'imgs')

        self.imgs_names = read_csv(csv_path)[0]

        img_size = 224
        ''' set up image transform '''
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        img = Image.open(os.path.join(self.img_dir, self.imgs_names[idx])).convert('RGB')
        images = self.transform(img)

        ''' read image '''
        return images, self.imgs_names[idx]