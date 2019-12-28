import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        vec_length = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(vec_length, 2048)

    def forward(self, im):
        x = self.resnet50(im)
        return x


