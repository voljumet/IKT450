
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import random
import torch
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import random
numpy.random.seed(7)


#Simple food classification example

#words
pizza = [1,0,0,0,0,0]
taco =  [0,1,0,0,0,0]
sushi = [0,0,1,0,0,0]
car =   [0,0,0,1,0,0]
bike =  [0,0,0,0,1,0]
truck = [0,0,0,0,0,1]

classone = [pizza,taco,sushi]
classtwo = [car,bike,truck]

x_train = torch.Tensor(classone+classtwo)
y_train = torch.Tensor([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])
x_test = x_train
y_test = y_train
print(x_train)
print(y_train)


#Variant 1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,256)
        self.fc2 = nn.Linear(256,2)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x



max_words = 6
num_classes = 2
batch_size = 1
epochs = 5

n = Net()
print("Model",n)
print("Parameters",[param.nelement() for param in n.parameters()])

from torchsummary import summary
summary(n,x_train.shape)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(n.parameters(), lr=0.001)#, momentum=0.9)
#optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

t_loss = []
v_loss = []

t_acc = []
v_acc = []

def avg(l):
    return sum(l)/len(l)
for i in range(3000):
    #n.train()
    y_pred_train = n(x_train)
    loss_train = loss_fn(y_pred_train,y_train)
    y_pred_test = n(x_test)
    loss_test = loss_fn(y_pred_test,y_test)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    if(i%25==0):
        print(loss_train.detach().numpy())



sentence_test1 = "I like pizza, taco and sushi"
sentence_test1_vector = [1,1,1,0,0,0]
sentence_test2 = "I have a car, bike and a truck"
sentence_test2_vector = [0,0,0,1,1,1]

def classify(vector):
    aindex = np.argmax(n(torch.Tensor(vector)).detach().numpy())
    if(aindex==0):
        print(vector,"Food")
    else:
        print(vector,"Cars")


classify(sentence_test1_vector)
classify(sentence_test2_vector)


