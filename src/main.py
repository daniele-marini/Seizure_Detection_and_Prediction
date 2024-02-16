import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from torch.utils.data import TensorDataset,DataLoader
from torch.utils import tensorboard

import mne
import os
import numpy as np
import random
from matplotlib import pyplot as plt

from typing import Callable, Dict, List, Tuple, Union
from timeit import default_timer as timer
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from utils import fix_random, process_file, create_test
from model import modified_resnet50
from traing_functions import EarlyStopper, training_loop
from evaluation_functions import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_random(42)

# load the data
patient_path = '/path/of/the/file.ecg'
recordings = os.listdir(patient_path)

test_recording = 'PN00-5.edf'
recordings.remove(test_recording)

info_train = {'PN00-1.edf':[1140,1215,2605],
              'PN00-2.edf':[1220,1275,2280],
              'PN00-3.edf':[765,825,2485],
              'PN00-4.edf':[1005,1080,2080]}

info_test = {'PN00-5.edf':[900,975,2120]}

# creation of the train dataloader
train_dataset = []

for recording in recordings[:3]:
  file_path = os.path.join(patient_path,recording)
  start,stop,end_recording = info_train[recording]
  data = process_file(file_path,start,stop,end_recording)
  train_dataset+=data

features = [torch.tensor(item[0]) for item in train_dataset]
labels = [item[1] for item in train_dataset]

features_tensor = torch.stack(features)
labels_tensor = torch.tensor(labels).squeeze()

# compute quantile
quantile = 0
for i in range(len(features_tensor)):
  quantile+=torch.quantile(features_tensor[i],0.90)

threshold = quantile/len(features_tensor)

# Create a boolean mask where True indicates elements larger than the threshold
mask = features_tensor > threshold
# Replace elements larger than the threshold with zero
features_tensor[mask] = 0.0

# reshape
features_tensor = features_tensor.permute(0,2,1,3)

# Create a TensorDataset
dataset = TensorDataset(features_tensor, labels_tensor)

# Create a DataLoader
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# definition of the model
model = modified_resnet50()

# TRAINING

lr = 0.001
num_epochs = 15
log_interval = 20
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience = 3, min_delta = 0)

statistics = training_loop(num_epochs, optimizer, log_interval, model, train_dataloader,early_stopping=early_stopper)

# save the model
path = "/content/drive/MyDrive/EEG_models"
if not os.path.exists(path):
  os.makedirs(path)

dir=os.path.join(path,'model.pdh')
torch.save(model,dir)

# load the model
# model = torch.load('/content/drive/MyDrive/EEG_models/model.pdh',map_location=torch.device(device))

# TESTING

# Preparation of the test dataloader
test_dataset = []

file_path = os.path.join(patient_path,'PN00-5.edf')
start,stop,end_seizure = info_test['PN00-5.edf']
data = create_test(file_path,start,stop,end_seizure)
test_dataset+=data

test_features = [item[0] for item in test_dataset]
test_labels = [item[1] for item in test_dataset]
test_features_tensor = torch.tensor(test_features)
labels_tensor = torch.tensor(test_labels).squeeze()

test_features_tensor = test_features_tensor.permute(0,2,1,3)

# Create a TensorDataset
dataset = TensorDataset(test_features_tensor, labels_tensor)

# Create a DataLoader
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

accuracy,errors_summary,FP,time_before = evaluate(model,test_dataloader,device)

start_seizure = info_test['PN00-5.edf'][0]
print(f'Accuracy --> {accuracy}')
print(f'Error Summary --> {errors_summary}')
print(f'False Positive Samples --> {FP}')
print(f'First Detection {start_seizure-time_before} seconds before the start of the seizure')