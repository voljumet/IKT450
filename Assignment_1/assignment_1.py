import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
dataset = numpy.loadtxt("/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/UiA/IKT450 - DNN/Assignments/Assignment_1/pima-indians-diabetes_data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8
# print(dataset[0:3, 8])

# split into input (X) and output (Y) variables
X_train = dataset[ :int(len(dataset)*splitratio), 0:8]
#                [0:80%, 0:8]

X_val = dataset[int(len(dataset)*splitratio):, 0:8]
#                [80%:, 0:8]

Y_train = dataset[:int(len(dataset)*splitratio), 8]
#                [0:80%, 8]

Y_val = dataset[int(len(dataset)*splitratio):, 8]
#                [20%:, 8]
# print(X_train)
# print(Y_train)
# x = X_val[:1, 0:8]
# y = X_train[1:2, 0:8]
# print("X: ", x,", x_train: ", y)
# print("result: ", x-y)
# print("ress: ", numpy.linalg.norm(x-y))



# Choose k-value here: _____________________________________
k_value = 30
#___________________________________________________________

def distance(one, two):
    return numpy.linalg.norm(one - two)


def shortest_distance(x, x_train, y_train):
    num_list_final = []
    num_list_return = []
    for i in range(len(x_train)):
        num_list_final.append((distance(x, x_train[i]), x_train[i], y_train[i]))
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

    return predicted


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(X_val)):
    # x = 
    y = Y_val[i]
    pred = shortest_distance(X_val[i], X_train, Y_train)
    print("Y:", pred, "Y-hat", y)
    # , "Distance:", shortest)

    if y == 1 and pred == 1:
        TP += 1

    if y == 0 and pred == 0:
        TN += 1

    if y == 1 and pred == 0:
        FN += 1

    if y == 0 and pred == 1:
        FP += 1

print("k-value", k_value)
print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))
print("Recall", TP/(TP+FN))
print("Precision", TP/(TP+FP))
print("F1", (2*TP)/(2*TP+FP+FN))

