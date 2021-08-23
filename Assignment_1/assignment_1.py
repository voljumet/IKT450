import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio), 0:8]
X_val = dataset[int(len(dataset)*splitratio):, 0:8]
Y_train = dataset[:int(len(dataset)*splitratio), 8]
Y_val = dataset[int(len(dataset)*splitratio):, 8]
print(X_train)
print(Y_train)
k_value = 80


def distance(one, two):
    return numpy.linalg.norm(one - two)


def shortest_distance(x, x_rest, y_rest):
    num_list_final = []
    num_list_return = []
    for i in range(len(x_rest)):
        num_list_final.append((distance(x, x_rest[i]), x_rest[i], y_rest[i]))
    num_list_final.sort()

    # fjernet iffen som sjekker om ny distance av lavere,
    # og appender alle avstander slik at listene appendes med samme funksjon under!

    if len(num_list_final) >= k_value:
        for k in range(k_value):
            num_list_return.append(num_list_final[k])

    varr = 0
    for f in range(len(num_list_return)):
        varr += num_list_return[f][2]

    if varr / k_value >= 0.5:
        predicted = 1.0
    else:
        predicted = 0.0

    return predicted, num_list_return


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    pred, shortest = shortest_distance(x, X_train, Y_train)
    print("Y:", pred, "Y hat", y)
          #, "Distance:", shortest)

    if y == 1 and pred == 1:
        TP += 1

    if y == 0 and pred == 0:
        TN += 1

    if y == 1 and pred == 0:
        FN += 1

    if y == 0 and pred == 1:
        FP += 1

print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))
print("Recall", TP/(TP+FN))
print("Precision", TP/(TP+FP))
print("F1", (2*TP)/(2*TP+FP+FN))

