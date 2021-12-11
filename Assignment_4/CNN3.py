import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from numpy import size, shape
from torch.utils.data import DataLoader
import pathlib
import cv2

def imshow(img, s=""):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if s and not '\n' in s:
        s = ' '.join(s.split())
        p = s.find(' ', int(len(s) / 2))
        s = s[:p] + "\n" + s[p + 1:]
    plt.text(0, -20, s)
    plt.show()


import torch.nn as nn
import torch.nn.functional as F

# this is the fucnton to calculate_H_And_W to feed in line 49 , 55
def calculate_H_And_W(hight):
    #padding is zero, stride is 1,
    #1 # conv
    print("before: ",  hight)
    H_out = ((hight + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    print("first: ", H_out)
    #2, stride is 2, kernel_size is 2, Pool
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
    #3 conv2d
    print("second: ", H_out)
    H_out = ((H_out + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    #4 MaxPool2d
    print("third: ", H_out)
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
    print("forth: ", H_out)
    print(H_out)


calculate_H_And_W(150)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1)


        self.fc1 = nn.Linear(in_features= 32 * 36 * 36, out_features= 120)
        self.fc2 = nn.Linear(120, 84)
        # 11 classes
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
       # print("x1: ",x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print("x2: ",x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("x3: ",x.shape)
        x = x.view(-1, 32 * 36 * 36)
        #print("x4: ",x.shape)
        x = F.relu(self.fc1(x))
        #print("x5: ",x.shape)
        x = F.relu(self.fc2(x))
        #print("x6: ", x.shape)
        x = self.fc3(x)
        #print("x7: ",x.shape)
        return x


import os.path as osp

def train():
    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                             [0.5, 0.5, 0.5])
    ])

    train_path = '/Users/peshangalo/Documents/Master/First_Year/DNN/CNN/archive_2/training'
    test_path = '/Users/peshangalo/Documents/Master/First_Year/DNN/CNN/archive_2/validation'

    trainloader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=64, shuffle=True
    )
    testloader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=32, shuffle=True
    )

    #root = pathlib.Path(train_path)
    #classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    classes = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit')
    print(classes)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    s = ' '.join('%5s' % classes[labels[j]] for j in range(32))
    print(s)
    imshow(torchvision.utils.make_grid(images), s)
    print(1)

    net = Net()
    # import torchvision.models as models
    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512,10)
    # import pdb;pdb.set_trace()
    import torch.optim as optim
    #
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i % 500 == 0):
                print(epoch, i, running_loss / (i + 1))

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    s1 = "Pred:" + ' '.join('%5s' % classes[predicted[j]] for j in range(32))
    s2 = "Actual:" + ' '.join('%5s' % classes[labels[j]] for j in range(32))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(32)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))
    imshow(torchvision.utils.make_grid(images), s1 + "\n" + s2)


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()
    train()
