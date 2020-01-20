import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of the net
import torch.nn.init as I
from torchvision.models import resnet50

class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.model = resnet50(pretrained=True)
        #changing first layer from to handle gray scale image (1 channel) instead of 3 channels
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #change last layer to ouput 136 values (2 for each point)
        self.model.fc = nn.Sequential(nn.Linear(2048, 1000),
                                    nn.ReLU(), 
                                    nn.Linear(1000, 136))

    def forward(self, x):
        x = self.model(x)
        return x