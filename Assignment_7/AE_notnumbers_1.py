
import numpy as np
import random
import matplotlib.pyplot as plt

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


input_dim = 784 #28*28
encoding_dim = 32#int(784/2)  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats




x_train = []
x_test = []

for i in range(10):
        x_train += [[float(random.randint(0,1)) for i in range(input_dim)]]*10
random.shuffle(x_train)

#x_train = np.array(x_train)
#x_test = np.array(x_test)





x_test = np.array([[float(random.randint(0,1)) for i in range(input_dim)]]*10000)


class EncNet(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fc1 = nn.Linear(input_dim,encoding_dim)
		#self.fc2 = nn.Linear(encoding_dim,input_dim)


	def forward(self,x):
		x = self.fc1(x)
		x = F.sigmoid(x)
		#x = self.fc2(x)
		#x = F.sigmoid(x)
		return x	




class DecNet(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		#self.fc1 = nn.Linear(input_dim,encoding_dim)
		self.fc2 = nn.Linear(encoding_dim,input_dim)


	def forward(self,x):
		#x = self.fc1(x)
		#x = F.sigmoid(x)
		x = self.fc2(x)
		x = F.sigmoid(x)
		return x	



#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.fc1 = nn.Linear(input_dim,encoding_dim)
#        self.fc2 = nn.Linear(encoding_dim,input_dim)
#
#    def forward(self,x):
#        x = self.fc1(x)
#        x = F.sigmoid(x)
#        x = self.fc2(x)
#        x = torch.sigmoid(x)
#        return x


#n = Net()

enc = EncNet()
dec = DecNet()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(enc.parameters())+list(dec.parameters()))
#optimizer = optim.SGD(n.parameters(), lr=0.001)#, momentum=0.9)
#optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for i in range(2000):
    y_pred = dec(enc(torch.tensor(x_train)))
    #y_pred = n(torch.tensor(x_train))
    loss_train = loss_fn(y_pred,torch.tensor(x_train))
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print(i)



encoded_img = enc(torch.tensor(x_train))
decoded_img = dec(encoded_img)

#decoded_img = enc(dec(torch.tensor(x_train))


n = 10  # how many digits we will display
plt.figure(figsize=(30, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(np.array(x_train[i]).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(np.array(encoded_img[i].detach().numpy()).reshape(16, 2))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n*2)
    plt.imshow(np.array(decoded_img[i].detach().numpy()).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

import sys
sys.exit(0)
