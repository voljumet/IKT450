import glob
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
#Dispaly images using matplotlib
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

# this is the function to calculate_H_And_W to feed in line 49 , 55
def calculate_H_And_W(hight):
    #padding is zero, stride is 1,
    #1  conv2d
    H_out = ((hight + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    #2 MaxPool2d, stride is 2, kernel_size is 2, Pool
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
    #3 conv2d
    H_out = ((H_out + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    #4 MaxPool2d
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 36 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# The main function that contain the training process and the evaluation process
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

    # Count number of images in both training and validation sets
    train_count = len(glob.glob(train_path + '/**/*.jpg'))
    test_count = len(glob.glob(test_path + '/**/*.jpg'))

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

    # iter() iterates over all elements in the trainloader
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
    train_accuracy = 0.0
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            # Compare prediction with labels and sum the matches
            train_accuracy += int(torch.sum(prediction == labels.data))
            if (i % 500 == 0):
                print(epoch, i, running_loss / (i + 1))
            test_accuracy = 0.0
            for i, (images, labels) in enumerate(testloader):
                 if torch.cuda.is_available():
                      images, labels = data
            outputs = net(images)
            _, prediction = torch.max(outputs.data, 1)
            # Sum number of matches through test set
            test_accuracy += int(torch.sum(prediction == labels.data))

            test_accuracy = test_accuracy / test_count
            train_accuracy = train_accuracy / train_count
            train_loss = running_loss / train_count
            print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

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
