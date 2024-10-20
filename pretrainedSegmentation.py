import torch
import torchvision.transforms
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
import torch.optim as optim
import warnings
from dataset import getMarkerDataset
from markerlab import JapanPattern
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Suppress FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


klein = 'Klein'
#getMarkerDataset(pattern=JapanPattern(), klein=klein)
Input = torch.load("Datasets//Input" + klein)
Output = torch.load("Datasets//Output" + klein)
binary_masked = torch.load("Datasets//binary" + klein)

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class customDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        image = image - torch.min(image)
        image = image/torch.max(image)
        output = self.mask_list[idx]
        no_class = torch.ones(size=output.shape, dtype=output.dtype)
        no_class = no_class - output
        #output = torch.dstack(tensors=(no_class, output))
        image = image.view(image.shape[::-1])
        output = output.view(output.shape[::-1])
        if self.transform:
            image = self.transform(image)
        return image, output

def getImage(x):
    x = x.detach().cpu().view(x.shape[2], x.shape[1], x.shape[0]).numpy()
    return np.astype(x/np.max(x)*255, int)

def getMask(x):
    x = x.detach().cpu().view(x.shape[::-1]).numpy()
    return np.astype(np.round(x), int)

print("CUDA: ", torch.cuda.is_available())
device = 'cuda'

dataset = customDataset(image_list=Input, mask_list=binary_masked, transform=normalize)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)#weights='DEFAULT')#pretrained=True)
#model.classifier[4] = nn.Conv2d(256, 2, 1)
#model.aux_classifier[4] = nn.Conv2d(10, 2, 1)
model.classifier = DeepLabHead(960, 2)

model = model.to(device)
weights = torch.tensor([99/100, 1/100]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr = 2 * 1e-4)
num_epochs = 25
"""
fig = plt.figure()
ax0 = fig.add_subplot(1, 4, 1)
ax1 = fig.add_subplot(1, 4, 2)
ax2 = fig.add_subplot(1, 4, 3)
ax3 = fig.add_subplot(1, 4, 4)"""
print("ok")
i = -1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in data_loader:
        i += 1
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        gradients = torch.abs(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))
        print(f'{i * data_loader.batch_size / dataset.__len__()*100} '
              f'mean gradient: {torch.mean(gradients)} max: {torch.max(gradients)}')
        optimizer.step()
        running_loss += loss.item()
        if i > 50:
            i = 0
            outImage = outputs.cpu().detach().numpy()[0]
            outImage = outImage.transpose(2, 1, 0)[:, :, 0]
            plt.imshow(outImage, cmap='Greys_r')
            plt.show()
            #torch.save(model.state_dict(), "models//pretrained")



    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}", outputs.shape, masks.shape)


    """o = outputs.data.cpu()[-1][0]
    o = torch.squeeze(o).numpy()
    ax1.imshow(np.transpose(o, (1,0)), cmap='Greys_r')
    o = outputs.data.cpu()[-1][1]
    o = torch.squeeze(o).numpy()
    ax0.imshow(np.transpose(o, (1,0)), cmap='Greys_r')
    ax2.imshow(getMask(masks[-1]), cmap='Greys_r')
    ax3.imshow(getImage(images[-1]))
    plt.show(block=False)
    plt.pause(5)
    plt.cla()"""

print("Training complete.")

"""
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

"""











