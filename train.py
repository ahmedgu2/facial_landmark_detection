from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets.data_transforms import Rescale, Normalize, ToTensor
from datasets.facialKeypointsDataset import FacialKeypointsDataset
from config import Config
import torch.optim as optim
import torch.nn as nn
from model.naimishnet import NaimishNet
from model.resnet import Resnet
import torch

def train():

    data_transform = transforms.Compose([Rescale((224, 224)),
                                        Normalize(),
                                        ToTensor()])
    #define train dataset
    train_data = FacialKeypointsDataset(Config().TRAINSET_CSV, 
                                    Config().TRAINGSET_DIR, 
                                    data_transform)
    #define train dataloader
    train_loader = DataLoader(train_data, 
                          batch_size=Config().batch_size,
                          shuffle=True)
    #define valid dataset
    valid_data = FacialKeypointsDataset(Config().TESTSET_CSV, 
                                    Config().TESTSET_DIR, 
                                    data_transform)
    #define valid dataloader
    valid_loader = DataLoader(valid_data, 
                          batch_size=Config().batch_size,
                          shuffle=True)
    
    #define model
    net = Resnet()
    net = net.cuda()

    #loss function
    criterion = nn.MSELoss()
    #optimzer
    optimizer = optim.Adam(net.parameters(), Config().learning_rate)

    best_loss = 100000

    for epoch in range(Config().epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_loss = 0.0

        print("Training Begins")
        net.train()

        # train on batches of data
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image'].cuda()
            key_pts = data['keypoints'].cuda()

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
        
        train_loss = running_loss/len(train_loader)
        print('Training : Epoch {}: Loss = {}'.format(epoch+1, train_loss))
        running_loss = 0.0
        
        print("Validation Begins")
        net.eval()
        with torch.no_grad():
            for batch_i, data in enumerate(valid_loader):
                # get the input images and their corresponding labels
                images = data['image'].cuda()
                key_pts = data['keypoints'].cuda()

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)

                # forward pass to get outputs
                output_pts = net(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)

                # print loss statistics
                running_loss += loss.item()
            
        valid_loss = running_loss/len(valid_loader)
        #save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), Config().MODEL_DIR + "model-{}-{:.4f}.model".format(epoch+1, valid_loss))
        
        print('Validation : Epoch {}: Loss = {}'.format(epoch+1, valid_loss))
        running_loss = 0.0

    print('Finished Training.\n')


if __name__ == "__main__":
    train()
    