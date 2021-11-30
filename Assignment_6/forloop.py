
categories = ["alex","ole","peshang"]
y_train_temp = ["alex","ole","peshang","alex","ole","peshang"]
num_classes = 30
y_train = []

for n in [categories.index(i) for i in y_train_temp]:
    y_train.append([0 for i in range(num_classes)])
    y_train[-1][n] = 1
    print(y_train)
    pass