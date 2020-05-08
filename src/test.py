'''
USAGE:
python test.py --model ../outputs/sports.pth --label-bin ../outputs/lb.pkl --input ../input/example_clips/lifting.mp4 --output ../outputs/lifting.mp4
'''

import torch
import numpy as np
import argparse
import joblib
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn_models
import albumentations

from torchvision.transforms import transforms   
from torch.utils.data import Dataset, DataLoader
from torchvision import models as models

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True, type=str,
	help="path to our output video")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("Loading model and label binarizer...")
lb = joblib.load(args["label_bin"])

model = cnn_models.CustomCNN().cuda()
print('Model Loaded...')

model.load_state_dict(torch.load(args["model"]))
print('Loaded model state_dict...')

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    # albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True)
    ])

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter(str(args['output']), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        model.eval()
        with torch.no_grad():
            image = frame
            image = aug(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float).cuda()
            image = image.unsqueeze(0)
            
            outputs = model(image)
            print(outputs)
            _, preds = torch.max(outputs.data, 1)
            print('PREDS', preds)
            print(f"Predicted output: {lb.classes_[preds]}")
        
        cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('image', frame)
        out.write(frame)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    else: 
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()