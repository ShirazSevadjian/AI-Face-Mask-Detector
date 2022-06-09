import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

#load
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = ImageFolder(root='./dataset', transform=transform)
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches +=1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5
    return mean, std

mean, std = get_mean_std(loader)

print(mean)
print(std)

