import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.image import imread
from dataset import getMarkerDataset
from torch.utils.data import Dataset, DataLoader
from markerlab import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
klein = 'Klein'
#getMarkerDataset(pattern=JapanPattern(), klein=klein)
Input = torch.load("Datasets//Input" + klein)
Output = torch.load("Datasets//Output" + klein)
print(Input[0].shape, Output[0].shape)

transform = transforms.Compose([
    #transforms.ToTensor(),  # Converts [0, 255] range to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=25, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=7, kernel_size=25, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=23, stride=1, padding='same', bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
class imgLoader(Dataset):
    def __init__(self, image_list, output_list, transform=None):
        self.image_list = image_list
        self.output_list = output_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #plt.imshow(self.output_list[idx])
        #plt.show()
        image = self.image_list[idx]/255
        output = self.output_list[idx]/255
        if self.transform:
            image = self.transform(image)
        return image.view(image.shape[2], image.shape[1], image.shape[0]), \
            output.view(output.shape[2], output.shape[1], output.shape[0])

def getImage(x):
    x = x.detach().cpu().view(x.shape[2], x.shape[1], x.shape[0]).numpy()
    return np.astype(x/np.max(x)*255, int)

print("CUDA: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = imgLoader(image_list=Input, output_list=Output)#, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

kernel = Kernel().to(device)

pos_weight = torch.tensor([1000], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#loss_fn = nn.L1Loss()
optimizer = optim.SGD(kernel.parameters(), lr=0.0001, momentum=0.9)
k = 0
for epoch in range(100):
    running_loss = 0.0
    k += 1
    for img, sol in data_loader:
        img = img.to(device)
        sol = sol.to(device)

        optimizer.zero_grad()
        out = kernel(img)
        loss = loss_fn(out, sol)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}], Loss: {running_loss / len(Input):.6f}, kernel weight:{kernel.conv3.weight[2][2][2]}"
          f"max : {torch.max(out[-1])}")
    if not k%10:
        ou = getImage(out[-1])
        so = getImage(sol[-1])
        im = getImage(img[-1])
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(ou)
        fig.add_subplot(1, 3, 2)
        plt.imshow(so)
        fig.add_subplot(1, 3, 3)
        plt.imshow(im)
        plt.show()
ou = getImage(out[-1])
so = getImage(sol[-1])
im = getImage(img[-1])
fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(ou)
fig.add_subplot(1, 3, 2)
plt.imshow(so)
fig.add_subplot(1, 3, 3)
plt.imshow(im)
plt.show()















