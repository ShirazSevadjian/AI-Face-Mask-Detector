import torch
import  torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

classes =["Cloth","N95","NoMask","Surgical"]

model = torch.load('./models/model60.pth')

image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4994, 0.4647, 0.4447), (0.2972, 0.2887, 0.2926))])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print("In the Image given the mask worn is an",classes[predicted.item()],"mask.")

classify(model, image_transforms, './singleImages/Indi1.jpg', classes)

img = Image.open('./singleImages/Indi1.jpg')
img.show()