import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of the net
import torch.nn.init as I


#The architeture below is based on NaimishNet (https://arxiv.org/pdf/1710.00977.pdf) with little variations.
class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## The last layer outputs 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(36864, 1024)
        self.drop5 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.drop6 = nn.Dropout2d(0.6)
        self.fc3 = nn.Linear(512, 136)

        
    def forward(self, x):
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x
