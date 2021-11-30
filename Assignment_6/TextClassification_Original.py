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


tempdata = open("movie_lines.txt", errors='replace').readlines()
x_train_temp = np.array([i.split("+++$+++")[-1].strip() for i in tempdata[:1000]])
y_train_temp = [i.split("+++$+++")[-2].strip() for i in tempdata[:1000]]

#import pdb;pdb.set_trace()
#Getting Categories

categories = list(set(y_train_temp))
y_train_org = np.array([categories.index(i) for i in y_train_temp])
x_train_org = x_train_temp[:]
print("Categories:"+str(categories))
num_classes = 30
y_train = []
for n in [categories.index(i) for i in y_train_temp]:
    y_train.append([0 for i in range(num_classes)])
    y_train[-1][n] = 1
    
print(y_train)
print(num_classes, 'classes')
#import pdb;pdb.set_trace()
#Embeddings
allwords = ' '.join(x_train_temp).lower().split(' ')
uniquewords = list(set(allwords))
#Stemming
from nltk.stem import *
stemmer = PorterStemmer()

x_train = []

def makeTextIntoNumbers(text):
    iwords = text.lower().split(' ')
    numbers = []
    for n in iwords:
        try:
            numbers.append(uniquewords.index(n))
        except ValueError:
            numbers.append(0)
    numbers = numbers + [0,0,0,0,0]

    return numbers[:6]

for i in x_train_temp:
    t = makeTextIntoNumbers(i)
    x_train.append(t)

print(set([len(i) for i in x_train]))
x_train = torch.LongTensor(x_train)
y_train = torch.Tensor(y_train)

# # Variant 1
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(6,256)
#         self.fc2 = nn.Linear(256,num_classes)
#
#     def forward(self,x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
#         return x

# Variant 2
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.embedding = nn.Embedding(len(uniquewords), 20)
        
        self.lstm = nn.LSTM(input_size=20,
                            hidden_size=16,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,inp):
        e = self.embedding(inp)
        output, hidden = self.lstm(e)

        x = self.fc1(output[:, -1, :])
        x = F.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

max_words = 6
batch_size = 1
epochs = 5

n = Net2()
print("Model",n)
print("Parameters",[param.nelement() for param in n.parameters()])

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(n.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

t_loss = []
v_loss = []

t_acc = []
v_acc = []

def avg(l):
    return sum(l)/len(l)

n_steps = 3000
for i in range(n_steps):
    y_pred_train = n(x_train)
    loss_train = loss_fn(y_pred_train, y_train)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    if (i % 25) == 0:
        print(loss_train.detach().numpy())


def classify(line):
    indices = makeTextIntoNumbers(line)
    tensor = torch.LongTensor([indices])
    output = n(tensor).detach().numpy()
    aindex = np.argmax(output)
    return aindex

def getRandomTextFromIndex(aIndex):
    res = -1
    while res != aIndex:
        aNumber = random.randint(0, len(y_train_org) - 1)
        res = y_train_org[aNumber]
    return x_train_org[aNumber] 

print("ready")
s = " "
while s:
    category = classify(s)
    print("Movie",category)
    text = getRandomTextFromIndex(category)
    print("Chatbot:" + text)
    s = input("Human:")

for line in x_train_temp[10:]:
    classify(line)

