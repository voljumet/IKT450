import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import random
numpy.random.seed(7)

#dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
dataset = numpy.loadtxt("crx.data.csv", delimiter=",", usecols=[1,2,3,8,11,14,15])

#import pdb;pdb.set_trace()

numpy.random.shuffle(dataset)
splitratio = 0.9



# split into input (X) and output (Y) variables
X_train = torch.Tensor(dataset[:int(len(dataset)*splitratio),0:6])
X_test = torch.Tensor(dataset[int(len(dataset)*splitratio):,0:6])
Y_train = torch.Tensor(dataset[:int(len(dataset)*splitratio),6])
Y_test = torch.Tensor(dataset[int(len(dataset)*splitratio):,6])
print(Y_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,6)
        self.fc2 = nn.Linear(6,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

n = Net()
print("Model",n)
print("Parameters",[param.nelement() for param in n.parameters()])

from torchsummary import summary
summary(n,X_train.shape)

#import pdb;pdb.set_trace()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(n.parameters(), lr=0.001)#, momentum=0.9)
optimizer = torch.optim.Adam(n.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

t_loss = []
v_loss = []

t_acc = []
v_acc = []

def avg(l):
    return sum(l)/len(l)
for i in range(300):
    #n.train()
    y_pred_train = n(X_train)
    loss_train = loss_fn(y_pred_train,Y_train)
    y_pred_test = n(X_test)
    loss_test = loss_fn(y_pred_test,Y_test)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    #n.eval()
    if(i%10==0):
        print(i,loss_train.item(),loss_test.item())#,loss_test.item())
    t_loss.append(loss_train.item())
    v_loss.append(loss_test.item())

    predictions_test = [round(i[0]) for i in y_pred_test.tolist()]
    accuracy_test = Y_test.tolist()
    acc_test = avg([abs(predictions_test[i]-accuracy_test[i]) for i in range(len(accuracy_test))])

    predictions_train = [round(i[0]) for i in y_pred_train.tolist()]
    accuracy_train = Y_train.tolist()
    acc_train = avg([abs(predictions_train[i]-accuracy_train[i]) for i in range(len(accuracy_train))])
    t_acc.append(acc_test)
    v_acc.append(acc_train)



import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(t_loss)
plt.plot(v_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(t_acc)
plt.plot(v_acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()






