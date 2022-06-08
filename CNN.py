# ----- Imports -----
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from skorch.helper import SliceDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from skorch import NeuralNetClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from FaceMaskCNN import FaceMaskCNN


# ----- Variables & Parameters -----
num_epochs = 4
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
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4994, 0.4647, 0.4447), (0.2972, 0.2887, 0.2926))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
train_loader = DataLoader(training_set, batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size, shuffle=False)


y_train = np.array([y for (x, y) in iter(training_set)])


model = FaceMaskCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('----- Starting training... -----')
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

print('----- Starting testing... -----')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 400 test images: {} %'
          .format((correct / total) * 100))


print('----- Saving model -----')
torch.save(FaceMaskCNN(), "./models/model4.pth")

