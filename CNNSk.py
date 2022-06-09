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
classes=('Cloth','N95','No Mask','Surgical')


# ----- Initialize the transformation configuration -----
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5211, 0.4858, 0.4651), (0.2889, 0.2824, 0.2880))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
train_loader = DataLoader(training_set, batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size, shuffle=False)


y_train = np.array([y for (x, y) in iter(training_set)])

torch.manual_seed(0)

net = NeuralNetClassifier(
    FaceMaskCNN,
    max_epochs=num_epochs,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr=1e-3,
    batch_size=batch_size,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device,
    )


print('----- Starting training... -----')

net.fit(training_set, y=y_train)

print('----- Training Complete -----')

print('----- Saving model -----')
torch.save(FaceMaskCNN(), "./models/model4sk.pth")

y_pred = net.predict(testing_set)
y_test = np.array([y for (x, y) in iter(testing_set)])
accuracy = accuracy_score(y_test, y_pred)
plot_confusion_matrix(net, testing_set, y_test.reshape(-1, 1),display_labels=classes)
plt.show()


precision, recall, fscore, support = score(y_test, y_pred)

print('accuracy: {}'.format(accuracy*100))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))