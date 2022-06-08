
import torch
from FaceMaskCNN import FaceMaskCNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


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
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
test_loader = DataLoader(testing_set, batch_size, shuffle=False)



model2 = torch.load('model2.pth')
print(model2)
model2.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'
        .format((correct / total) * 100))
