from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import os
import matplotlib.pyplot as plt
import numpy as np

num_epochs = 4
num_classes = 4
learning_rate = 0.001

#Transform the images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Train and Test datasets
dataset = torchvision.datasets.ImageFolder(root="./dataset", transform=transform)
trainset, testset = torch.utils.data.random_split(dataset, [1200, 400])

#Loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)



