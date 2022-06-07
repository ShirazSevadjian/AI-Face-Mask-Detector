# ----- Imports -----
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.datasets
import numpy as np


# ----- Variables & Parameters -----
num_epochs = 4
num_classes = 4
learning_rate = 0.001
dataset_root = "./dataset"
training_set_size = 1200
testing_set_size = 400
batch_size = 32

# ----- Initialize the transformation configuration -----
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
train_loader = DataLoader(training_set, batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size, shuffle=False)



