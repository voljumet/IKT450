import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import glob
from torch.optim import Adam
from torch.autograd import Variable
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
    # padding is zero, stride is 1,
    # 1 # conv
    H_out = ((hight + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    # 2, stride is 2, kernel_size is 2, Pool
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
    # 3 conv2d
    H_out = ((H_out + 2 * 0 - 1 * (3 - 1) - 1) / 1) + 1
    # 4 MaxPool2d
    H_out = ((H_out + 2 * 0 - 1 * (2 - 1) - 1) / 2) + 1
    print(H_out)
    return H_out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=32 * 36 * 36, out_features=120)
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

###############################################################################################

# Pre define the classes
classes = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup',
           'Vegetable-Fruit')
train_path = '/Users/peshangalo/Documents/Master/First_Year/DNN/CNN/archive_2/training'
validation_path = '/Users/peshangalo/Documents/Master/First_Year/DNN/CNN/archive_2/validation'

# Counting the number of images in both the training and validation path
train_count = len(glob.glob(train_path + '/**/*.jpg'))
validation_count = len(glob.glob(validation_path + '/**/*.jpg'))

# Run using the GPU if it is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Define image transformation, resize, flip and this will be applied on each element in the data set
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])
# Load the train and validation datasets, and make the shuffle true
trainloader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=32, shuffle=True
)
validationloader = DataLoader(
    torchvision.datasets.ImageFolder(validation_path, transform=transformer),
    batch_size=32, shuffle=True
)

model = Net().to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 100
best_accuracy = 0.0
for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    #########################Traning#############################
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))

    ######################################################
    # Evaluation on validation dataset
    model.eval()
    ###############validationing########################
    validation_accuracy = 0.0
    for i, (images, labels) in enumerate(validationloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        validation_accuracy += int(torch.sum(prediction == labels.data))
    ######################## Print out the values after each epoch  #########################

    train_accuracy = train_accuracy / train_count
    validation_accuracy = validation_accuracy / validation_count
    train_loss = train_loss / train_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' validation Accuracy: ' + str(validation_accuracy))

    # Save the best model
    if validation_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accurac = validation_accuracy

    #import the model
    #Use the same
