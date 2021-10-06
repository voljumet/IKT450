import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
dataset = numpy.loadtxt("/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/UiA/IKT450 - DNN/Assignments/Assignment_1/pima-indians-diabetes_data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
print(X_train)
print(Y_train)

def distance(one,two):
    return numpy.linalg.norm(one-two)

def shortestDistance(x,x_rest,y_rest):
    shortest = distance(x,x_rest[0])
    predicted = y_rest[0]
    for i in range(len(x_rest)):
        if distance(x,x_rest[i])<=shortest:
            shortest = distance(x,x_rest[i])
            predicted = y_rest[i]
    return predicted,shortest


TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    pred,shortest = shortestDistance(x,X_train,Y_train)
    print("Y:",pred,"Y hat",y,"Distance:",shortest)

    if(y==1 and pred ==1):
        TP += 1

    if(y==0 and pred ==0):
        TN += 1

    if(y==1 and pred ==0):
        FN += 1

    if(y==0 and pred ==1):
        FP += 1

print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
print("Recall",TP/(TP+FN))
print("Precision",TP/(TP+FP))
print("F1",(2*TP)/(2*TP+FP+FN))

