# ----- Imports -----
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
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
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
train_loader = DataLoader(training_set, batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size, shuffle=False)


y_train = np.array([y for (x, y) in iter(training_set)])


class FaceMaskCNN(nn.Module):

    def __init__(self):
        super(FaceMaskCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512,6)
            )


    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


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

y_pred = net.predict(testing_set)
y_test = np.array([y for (x, y) in iter(testing_set)])

accuracy = accuracy_score(y_test, y_pred)

labels = ['Cloth', 'N95', 'NoMask', 'Surgical']

plot_confusion_matrix(net, testing_set, y_test.reshape(-1, 1))

plt.show()


precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print('accuracy: {}'.format(accuracy))



print('----- Saving model -----')
torch.save(FaceMaskCNN().state_dict(),"model.pth")




# net.fit(training_set, y=y_train)
# train_sliceable = SliceDataset(training_set)
# scores = cross_val_score(net, train_sliceable, y_train, cv=5, scoring='accuracy')
