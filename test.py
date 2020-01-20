#%%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets.data_transforms import Rescale, Normalize, ToTensor
from config import Config
from model.naimishnet import NaimishNet
from model.resnet import Resnet
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from utils.utils import show_all_keypoints


def predict(img_url):

    #load image
    image = cv2.imread(img_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 2, 6)

    #load pretrained model
    model = Resnet()
    model.load_state_dict(torch.load("runs/model-28-0.0046.model"))

    #switch to evaluation mode
    model.eval()

    # loop over the detected faces from your haar cascade
    for i, (x,y,w,h) in enumerate(faces):

        # Select the region of interest that is the face in the image        
        roi = image[y-30:y+h+30, x-30:x+w+30]
        
        h, w = roi.shape[0], roi.shape[1]
        roi_disp = np.copy(roi)

        #Convert the face region from RGB to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        #Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi = roi / 255.0
        #Rescale the detected face to be the expected square size for the CNN
        roi = cv2.resize(roi, (224, 224)).reshape(224, 224, 1)
        #Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        roi = torch.from_numpy(roi.transpose(2, 0, 1))
        
        #Make facial keypoint predictions using the loaded, trained network         
        roi_copy = np.copy(roi)
        # convert images to FloatTensors
        roi = roi.type(torch.FloatTensor)
        roi = roi.unsqueeze(0) # convert to 4D tensor with batch size of 1
        # forward pass to get outputs
        output_pts = model.forward(Variable(roi, volatile=True))
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(68, 2)    
        # Convert back to numpy
        output_pts = output_pts.data.numpy()
        
        print("outputs_pts.shape:", output_pts.shape)
        
        #For displaying each detected face and the corresponding keypoints,
        #we need to undo the normalization and rescaling operations      
        roi = np.transpose(roi_copy, (1, 2, 0))
        # undo normalization of keypoints
        output_pts = (output_pts * 50.0) + 100
        # undo rescaling (to display on original roi image)
        output_pts = output_pts * (w / 224, h / 224)

        plt.figure(figsize=(10, 8))
        if (i == 0):
            plt.subplot(2, 1, 1)
            show_all_keypoints(np.squeeze(roi_disp), output_pts)
        elif (i==1):
            plt.subplot(2, 1, 2)
            show_all_keypoints(np.squeeze(roi_disp), output_pts)
        plt.show()


#if __name__ == "__main__":
predict('obamas.jpg')
# %%
