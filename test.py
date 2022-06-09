
import torch
import torch.nn as nn
from FaceMaskCNN import FaceMaskCNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from skorch.helper import SliceDataset
from skorch import NeuralNetClassifier
from sklearn.metrics import precision_recall_fscore_support as score


# ----- Variables & Parameters -----
num_epochs = 1
num_classes = 4
learning_rate = 0.001
dataset_root = "./dataset"
training_set_size = 1200
testing_set_size = 400
batch_size = 32
device = torch.device("cpu")
img_size = 64

# ----- Initialize the transformation configuration -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize((0.5211, 0.4858, 0.4651), (0.2889, 0.2824, 0.2880))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
test_loader = DataLoader(testing_set, batch_size, shuffle=False)



model2 = torch.load('./models/model60.pth')
model2.eval()
labelS =[]
predictions=[]
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        labelS.append(labels.tolist())
        predictions.append(predicted.tolist())


    print('Test Accuracy of the model on the 400 test images: {} %'
          .format((correct / total) * 100))

l = np.array(labelS)
p = np.array(predictions)

sl = sparse.csr_matrix(l)   # Here's the initialization of the sparse matrix.
sp = sparse.csr_matrix(p)




