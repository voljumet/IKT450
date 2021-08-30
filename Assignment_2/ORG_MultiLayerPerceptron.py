import torch
import torch.nn as nn
import torch.optim as optim


#crim	tax	rm	age	ptratio	medv
training_dataset0 = """
0.00632	296	6.575	65.2	15.3	24
0.02731	242	6.421	78.9	17.8	21.6
0.03237	222	6.998	45.8	18.7	33.4
0.06905	222	7.147	54.2	18.7	36.2
0.08829	311	6.012	66.6	15.2	22.9
0.22489	311	6.377	94.3	15.2	15
0.11747	311	6.009	82.9	15.2	18.9
0.09378	311	5.889	39	15.2	21.7
0.62976	307	5.949	61.8	21	20.4
"""
training_dataset = """
0.49  0.29  0.48  0.50  0.56  0.24  0.35  1
0.07  0.40  0.48  0.50  0.54  0.35  0.44  1
0.56  0.40  0.48  0.50  0.49  0.37  0.46  1
0.59  0.49  0.48  0.50  0.52  0.45  0.36  1
0.23  0.32  0.48  0.50  0.55  0.25  0.35  1
0.67  0.39  0.48  0.50  0.36  0.38  0.46  1
0.29  0.28  0.48  0.50  0.44  0.23  0.34  1
0.21  0.34  0.48  0.50  0.51  0.28  0.39  1
0.20  0.44  0.48  0.50  0.46  0.51  0.57  1
0.42  0.40  0.48  0.50  0.56  0.18  0.30  1
0.42  0.24  0.48  0.50  0.57  0.27  0.37  1
0.25  0.48  0.48  0.50  0.44  0.17  0.29  1
0.39  0.32  0.48  0.50  0.46  0.24  0.35  1
0.51  0.50  0.48  0.50  0.46  0.32  0.35  1
0.22  0.43  0.48  0.50  0.48  0.16  0.28  1
0.25  0.40  0.48  0.50  0.46  0.44  0.52  1
0.34  0.45  0.48  0.50  0.38  0.24  0.35  1
0.44  0.27  0.48  0.50  0.55  0.52  0.58  1
0.23  0.40  0.48  0.50  0.39  0.28  0.38  1
0.41  0.57  0.48  0.50  0.39  0.21  0.32  1
0.40  0.45  0.48  0.50  0.38  0.22  0.00  1
0.31  0.23  0.48  0.50  0.73  0.05  0.14  1
0.51  0.54  0.48  0.50  0.41  0.34  0.43  1
0.30  0.16  0.48  0.50  0.56  0.11  0.23  1
0.36  0.39  0.48  0.50  0.48  0.22  0.23  1
0.29  0.37  0.48  0.50  0.48  0.44  0.52  1
0.25  0.40  0.48  0.50  0.47  0.33  0.42  1
0.21  0.51  0.48  0.50  0.50  0.32  0.41  1
0.43  0.37  0.48  0.50  0.53  0.35  0.44  1
0.43  0.39  0.48  0.50  0.47  0.31  0.41  1
0.53  0.38  0.48  0.50  0.44  0.26  0.36  1
0.34  0.33  0.48  0.50  0.38  0.35  0.44  1
0.56  0.51  0.48  0.50  0.34  0.37  0.46  1
0.40  0.29  0.48  0.50  0.42  0.35  0.44  1
0.24  0.35  0.48  0.50  0.31  0.19  0.31  1
0.36  0.54  0.48  0.50  0.41  0.38  0.46  1
0.29  0.52  0.48  0.50  0.42  0.29  0.39  1
0.65  0.47  0.48  0.50  0.59  0.30  0.40  1
0.32  0.42  0.48  0.50  0.35  0.28  0.38  1
0.38  0.46  0.48  0.50  0.48  0.22  0.29  1
0.33  0.45  0.48  0.50  0.52  0.32  0.41  1
0.30  0.37  0.48  0.50  0.59  0.41  0.49  1
0.40  0.50  0.48  0.50  0.45  0.39  0.47  1
0.28  0.38  0.48  0.50  0.50  0.33  0.42  1
0.61  0.45  0.48  0.50  0.48  0.35  0.41  1
0.17  0.38  0.48  0.50  0.45  0.42  0.50  1
0.44  0.35  0.48  0.50  0.55  0.55  0.61  1
0.43  0.40  0.48  0.50  0.39  0.28  0.39  1
0.42  0.35  0.48  0.50  0.58  0.15  0.27  1
0.23  0.33  0.48  0.50  0.43  0.33  0.43  1
0.37  0.52  0.48  0.50  0.42  0.42  0.36  1
0.29  0.30  0.48  0.50  0.45  0.03  0.17  1
0.22  0.36  0.48  0.50  0.35  0.39  0.47  1
0.23  0.58  0.48  0.50  0.37  0.53  0.59  1
0.47  0.47  0.48  0.50  0.22  0.16  0.26  1
0.54  0.47  0.48  0.50  0.28  0.33  0.42  1
0.51  0.37  0.48  0.50  0.35  0.36  0.45  1
0.40  0.35  0.48  0.50  0.45  0.33  0.42  1
0.44  0.34  0.48  0.50  0.30  0.33  0.43  1
0.42  0.38  0.48  0.50  0.54  0.34  0.43  1
0.44  0.56  0.48  0.50  0.50  0.46  0.54  1
0.52  0.36  0.48  0.50  0.41  0.28  0.38  1
0.36  0.41  0.48  0.50  0.48  0.47  0.54  1
0.18  0.30  0.48  0.50  0.46  0.24  0.35  1
0.47  0.29  0.48  0.50  0.51  0.33  0.43  1
0.24  0.43  0.48  0.50  0.54  0.52  0.59  1
0.25  0.37  0.48  0.50  0.41  0.33  0.42  1
0.52  0.57  0.48  0.50  0.42  0.47  0.54  1
0.25  0.37  0.48  0.50  0.43  0.26  0.36  1
0.35  0.48  0.48  0.50  0.56  0.40  0.48  1
0.26  0.26  0.48  0.50  0.34  0.25  0.35  1
0.44  0.51  0.48  0.50  0.47  0.26  0.36  1
0.37  0.50  0.48  0.50  0.42  0.36  0.45  1
0.44  0.42  0.48  0.50  0.42  0.25  0.20  1
0.24  0.43  0.48  0.50  0.37  0.28  0.38  1
0.42  0.30  0.48  0.50  0.48  0.26  0.36  1
0.48  0.42  0.48  0.50  0.45  0.25  0.35  1
0.41  0.48  0.48  0.50  0.51  0.44  0.51  1
0.44  0.28  0.48  0.50  0.43  0.27  0.37  1
0.29  0.41  0.48  0.50  0.48  0.38  0.46  1
0.34  0.28  0.48  0.50  0.41  0.35  0.44  1
0.41  0.43  0.48  0.50  0.45  0.31  0.41  1
0.29  0.47  0.48  0.50  0.41  0.23  0.34  1
0.34  0.55  0.48  0.50  0.58  0.31  0.41  1
0.36  0.56  0.48  0.50  0.43  0.45  0.53  1
0.40  0.46  0.48  0.50  0.52  0.49  0.56  1
0.50  0.49  0.48  0.50  0.49  0.46  0.53  1
0.52  0.44  0.48  0.50  0.37  0.36  0.42  1
0.50  0.51  0.48  0.50  0.27  0.23  0.34  1
0.53  0.42  0.48  0.50  0.16  0.29  0.39  1
0.34  0.46  0.48  0.50  0.52  0.35  0.44  1
0.40  0.42  0.48  0.50  0.37  0.27  0.27  1
0.41  0.43  0.48  0.50  0.50  0.24  0.25  1
0.30  0.45  0.48  0.50  0.36  0.21  0.32  1
0.31  0.47  0.48  0.50  0.29  0.28  0.39  1
0.64  0.76  0.48  0.50  0.45  0.35  0.38  1
0.35  0.37  0.48  0.50  0.30  0.34  0.43  1
0.57  0.54  0.48  0.50  0.37  0.28  0.33  1
0.65  0.55  0.48  0.50  0.34  0.37  0.28  1
0.51  0.46  0.48  0.50  0.58  0.31  0.41  1
0.38  0.40  0.48  0.50  0.63  0.25  0.35  1
0.24  0.57  0.48  0.50  0.63  0.34  0.43  1
0.38  0.26  0.48  0.50  0.54  0.16  0.28  1
0.33  0.47  0.48  0.50  0.53  0.18  0.29  1
0.24  0.34  0.48  0.50  0.38  0.30  0.40  1
0.26  0.50  0.48  0.50  0.44  0.32  0.41  1
0.44  0.49  0.48  0.50  0.39  0.38  0.40  1
0.43  0.32  0.48  0.50  0.33  0.45  0.52  1
0.49  0.43  0.48  0.50  0.49  0.30  0.40  1
0.47  0.28  0.48  0.50  0.56  0.20  0.25  1
0.32  0.33  0.48  0.50  0.60  0.06  0.20  1
0.34  0.35  0.48  0.50  0.51  0.49  0.56  1
0.35  0.34  0.48  0.50  0.46  0.30  0.27  1
0.38  0.30  0.48  0.50  0.43  0.29  0.39  1
0.38  0.44  0.48  0.50  0.43  0.20  0.31  1
0.41  0.51  0.48  0.50  0.58  0.20  0.31  1
0.34  0.42  0.48  0.50  0.41  0.34  0.43  1
0.51  0.49  0.48  0.50  0.53  0.14  0.26  1
0.25  0.51  0.48  0.50  0.37  0.42  0.50  1
0.29  0.28  0.48  0.50  0.50  0.42  0.50  1
0.25  0.26  0.48  0.50  0.39  0.32  0.42  1
0.24  0.41  0.48  0.50  0.49  0.23  0.34  1
0.17  0.39  0.48  0.50  0.53  0.30  0.39  1
0.04  0.31  0.48  0.50  0.41  0.29  0.39  1
0.61  0.36  0.48  0.50  0.49  0.35  0.44  1
0.34  0.51  0.48  0.50  0.44  0.37  0.46  1
0.28  0.33  0.48  0.50  0.45  0.22  0.33  1
0.40  0.46  0.48  0.50  0.42  0.35  0.44  1
0.23  0.34  0.48  0.50  0.43  0.26  0.37  1
0.37  0.44  0.48  0.50  0.42  0.39  0.47  1
0.00  0.38  0.48  0.50  0.42  0.48  0.55  1
0.39  0.31  0.48  0.50  0.38  0.34  0.43  1
0.30  0.44  0.48  0.50  0.49  0.22  0.33  1
0.27  0.30  0.48  0.50  0.71  0.28  0.39  1
0.17  0.52  0.48  0.50  0.49  0.37  0.46  1
0.36  0.42  0.48  0.50  0.53  0.32  0.41  1
0.30  0.37  0.48  0.50  0.43  0.18  0.30  1
0.26  0.40  0.48  0.50  0.36  0.26  0.37  1
0.40  0.41  0.48  0.50  0.55  0.22  0.33  1
0.22  0.34  0.48  0.50  0.42  0.29  0.39  1
0.44  0.35  0.48  0.50  0.44  0.52  0.59  1
0.27  0.42  0.48  0.50  0.37  0.38  0.43  1
0.16  0.43  0.48  0.50  0.54  0.27  0.37  1
0.06  0.61  0.48  0.50  0.49  0.92  0.37  0
0.44  0.52  0.48  0.50  0.43  0.47  0.54  0
0.63  0.47  0.48  0.50  0.51  0.82  0.84  0
0.23  0.48  0.48  0.50  0.59  0.88  0.89  0
0.34  0.49  0.48  0.50  0.58  0.85  0.80  0
0.43  0.40  0.48  0.50  0.58  0.75  0.78  0
0.46  0.61  0.48  0.50  0.48  0.86  0.87  0
0.27  0.35  0.48  0.50  0.51  0.77  0.79  0
0.52  0.39  0.48  0.50  0.65  0.71  0.73  0
0.29  0.47  0.48  0.50  0.71  0.65  0.69  0
0.55  0.47  0.48  0.50  0.57  0.78  0.80  0
0.12  0.67  0.48  0.50  0.74  0.58  0.63  0
0.40  0.50  0.48  0.50  0.65  0.82  0.84  0
0.73  0.36  0.48  0.50  0.53  0.91  0.92  0
0.84  0.44  0.48  0.50  0.48  0.71  0.74  0
0.48  0.45  0.48  0.50  0.60  0.78  0.80  0
0.54  0.49  0.48  0.50  0.40  0.87  0.88  0
0.48  0.41  0.48  0.50  0.51  0.90  0.88  0
0.50  0.66  0.48  0.50  0.31  0.92  0.92  0
0.72  0.46  0.48  0.50  0.51  0.66  0.70  0
0.47  0.55  0.48  0.50  0.58  0.71  0.75  0
0.33  0.56  0.48  0.50  0.33  0.78  0.80  0
0.64  0.58  0.48  0.50  0.48  0.78  0.73  0
0.54  0.57  0.48  0.50  0.56  0.81  0.83  0
0.47  0.59  0.48  0.50  0.52  0.76  0.79  0
0.63  0.50  0.48  0.50  0.59  0.85  0.86  0
0.49  0.42  0.48  0.50  0.53  0.79  0.81  0
0.31  0.50  0.48  0.50  0.57  0.84  0.85  0
0.74  0.44  0.48  0.50  0.55  0.88  0.89  0
0.33  0.45  0.48  0.50  0.45  0.88  0.89  0
0.45  0.40  0.48  0.50  0.61  0.74  0.77  0
0.71  0.40  0.48  0.50  0.71  0.70  0.74  0
0.50  0.37  0.48  0.50  0.66  0.64  0.69  0
0.66  0.53  0.48  0.50  0.59  0.66  0.66  0
0.60  0.61  0.48  0.50  0.54  0.67  0.71  0
0.83  0.37  0.48  0.50  0.61  0.71  0.74  0
0.34  0.51  0.48  0.50  0.67  0.90  0.90  0
0.63  0.54  0.48  0.50  0.65  0.79  0.81  0
0.70  0.40  0.48  0.50  0.56  0.86  0.83  0
0.60  0.50  1.00  0.50  0.54  0.77  0.80  0
0.16  0.51  0.48  0.50  0.33  0.39  0.48  0
0.74  0.70  0.48  0.50  0.66  0.65  0.69  0
0.20  0.46  0.48  0.50  0.57  0.78  0.81  0
0.89  0.55  0.48  0.50  0.51  0.72  0.76  0
0.70  0.46  0.48  0.50  0.56  0.78  0.73  0
0.12  0.43  0.48  0.50  0.63  0.70  0.74  0
0.61  0.52  0.48  0.50  0.54  0.67  0.52  0
0.33  0.37  0.48  0.50  0.46  0.65  0.69  0
0.63  0.65  0.48  0.50  0.66  0.67  0.71  0
0.41  0.51  0.48  0.50  0.53  0.75  0.78  0
0.34  0.67  0.48  0.50  0.52  0.76  0.79  0
0.58  0.34  0.48  0.50  0.56  0.87  0.81  0
0.59  0.56  0.48  0.50  0.55  0.80  0.82  0
0.51  0.40  0.48  0.50  0.57  0.62  0.67  0
0.50  0.57  0.48  0.50  0.71  0.61  0.66  0
0.60  0.46  0.48  0.50  0.45  0.81  0.83  0
0.37  0.47  0.48  0.50  0.39  0.76  0.79  0
0.58  0.55  0.48  0.50  0.57  0.70  0.74  0
0.36  0.47  0.48  0.50  0.51  0.69  0.72  0
0.39  0.41  0.48  0.50  0.52  0.72  0.75  0
0.35  0.51  0.48  0.50  0.61  0.71  0.74  0
0.31  0.44  0.48  0.50  0.50  0.79  0.82  0
0.61  0.66  0.48  0.50  0.46  0.87  0.88  0
0.48  0.49  0.48  0.50  0.52  0.77  0.71  0
0.11  0.50  0.48  0.50  0.58  0.72  0.68  0
0.31  0.36  0.48  0.50  0.58  0.94  0.94  0
0.68  0.51  0.48  0.50  0.71  0.75  0.78  0
0.69  0.39  0.48  0.50  0.57  0.76  0.79  0
0.52  0.54  0.48  0.50  0.62  0.76  0.79  0
0.46  0.59  0.48  0.50  0.36  0.76  0.23  0
0.36  0.45  0.48  0.50  0.38  0.79  0.17  0
0.00  0.51  0.48  0.50  0.35  0.67  0.44  0
0.10  0.49  0.48  0.50  0.41  0.67  0.21  0
0.30  0.51  0.48  0.50  0.42  0.61  0.34  0
0.61  0.47  0.48  0.50  0.00  0.80  0.32  0
0.63  0.75  0.48  0.50  0.64  0.73  0.66  0
0.71  0.52  0.48  0.50  0.64  1.00  0.99  0
"""


training_dataset = [[float(f) for f in i.split("  ")] for i in training_dataset.strip().split("\n")]

#Making into a binary classifier
# training_dataset = [row[:-1]+[0 if row[-1] < 25 else 1] for row in training_dataset]
# training_dataset = training_dataset*1000

#print(training_dataset)
X = torch.Tensor([i[0:7] for i in training_dataset])
Y = torch.Tensor([i[7] for i in training_dataset])

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(7,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = Net()
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

allloss = []

for epoch in range(100):
    print(epoch)
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward()
    optimizer.step()
    allloss.append(loss.item())
import pdb;pdb.set_trace()


import matplotlib.pyplot as plt
plt.plot(allloss)
plt.show()

print(list(model.parameters()))

import sys
sys.exit(0)

#From scratch
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]


import math


def sigmoid(z):
    if( z <-100):
        return 0
    if( z >100):
        return 1
    return 1.0/math.exp(-z)


def firstLayer(row, weights):
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]

    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    return sigmoid(activation_1), sigmoid(activation_2)


def secondLayer(row, weights):
    activation_3 = weights[6]
    activation_3 += weights[7]*row[0]
    activation_3 += weights[8]*row[1]
    return sigmoid(activation_3)


def predict(row, weights):
    input_layer = row
    first_layer = firstLayer(input_layer, weights)
    second_layer = secondLayer(first_layer, weights)
    return second_layer, first_layer


for d in training_dataset:
    print(predict(d, weights)[0], d[-1])   #Prints y_hat and y


def train_weights(train, learningrate, epochs):
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            prediction, first_layer = predict(row, weights)
            error = row[-1]-prediction
            sum_error += error

            #First layer

            weights[0] = weights[0] + learningrate*error*1
            weights[1] = weights[1] + learningrate*error*row[0]
            weights[2] = weights[2] + learningrate*error*row[1]


            weights[3] = weights[3] + learningrate*error*1
            weights[4] = weights[4] + learningrate*error*row[2]
            weights[5] = weights[5] + learningrate*error*row[3]

            #Second layer
            weights[6] = weights[6] + learningrate*error
            weights[7] = weights[7] + learningrate*error*first_layer[0]
            weights[8] = weights[8] + learningrate*error*first_layer[1]
        if((epoch%100==0) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return weights


learningrate = 0.0001 #0.00001
epochs = 1000
train_weights = train_weights(training_dataset, learningrate, epochs)
print(train_weights)
