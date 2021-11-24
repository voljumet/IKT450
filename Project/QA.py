import glob
import json
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords

from cuda_check import check_if_cuda

# test = get_dataset_split_names('natural_questions')
# YesNoAnswer = dataset[0]['annotations']['yes_no_answer']
# Question = dataset[0]['question']
# AnswerasTokens = dataset[0]['document']['tokens']['token']

# check_if_cuda()

''' dev function to simply load from local file '''



def json_reader(path):
    for file in glob.glob(path, recursive=True):
        with open(file) as json_file:
            dataset.append(json.load(json_file))


dataset = []

json_reader('Data/mydata*.json')

''' --------------------------------'''
questions = []
short_answers = []
long_answer = []

def filter_html(data):
    long_answer_temp = []
    for i, each in enumerate(data[2]['is_html']):
        if each == 0:
            long_answer_temp.append(data[2]['token'][i])
    return long_answer_temp

for each in dataset:
    questions.append(each[0])
    short_answers.append(each[1])
    long_answer.append(filter_html(each))

    # for server:
    # questions.append(dataset['question']['tokens'])
    # short_answers.append(dataset['annotations']['yes_no_answer'])
    # long_answer_temp.append(dataset['document']['tokens'])



# remove all but short answer data


# uses 'is_html' to filer out html from the long answer

# for i, each in enumerate(long_answer_temp[0]['is_html']):
#     if each == 0:
#         long_answer.append(long_answer_temp[0]['token'][i])


print("Data read!")



stopwords = stopwords.words('english')
def remove_stopwords(dataset):
    return dataset.apply(lambda stop_word: ' '.join([word for word in stop_word.lower() if word not in stopwords]))


# remove_stopwords on data
for each_question in questions:
    for each_word in each_question:
        if each_word.lower() not in stopwords:
            pass


x_train_temp = remove_stopwords(questions)

categories = list(set(short_answers))

# does not need to be sorted if yes_no_answer is already 1 and 0
# y_train_org = np.array([categories.index(i) for i in y_train_temp_question])

allwords = ' '.join(x_train_temp).lower().split(' ')
unique_words = list(set(allwords))


y_train = short_answers

x_train = []

# take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
max_words = 30

def makeTextIntoNumbers(text):
    iwords = text.lower().split(' ')
    numbers = np.zeros(max_words, dtype=int)
    for count, each_word in enumerate(iwords):
        if count == max_words:
            break
        try:
            numbers[count] = unique_words.index(each_word)
        except ValueError:
            continue

    return list(numbers[:max_words])

x_train = torch.LongTensor(x_train)
y_train = torch.Tensor(y_train)

class Net(nn.Module):
    def __init__(self):
        super(nn, self).__init__()
        self.embedding = nn.Embedding(len(unique_words), 20)

        self.lstm = nn.LSTM(input_size=20,
                            hidden_size=10,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e = self.embedding(x)
        output, hidden = self.lstm(e)

        X = self.fc1(output[:, -1, :])
        X = F.relu(X)

        X = self.fc2(X)
        X = torch.sigmoid(X)

        return X

max_words = 6
batch_size = 1
epochs = 5

nene = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(nene.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


def avg(l):
    return sum(l)/len(l)

n_steps = 3000
for i in range(n_steps):
    y_pred_train = nene(x_train)
    loss_train = loss_fn(y_pred_train, y_train)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    if (i % 25) == 0:
        print(loss_train.detach().numpy())


def classify(line):
    indices = makeTextIntoNumbers(line)
    tensor = torch.LongTensor([indices])
    output = nene(tensor).detach().numpy()
    aindex = np.argmax(output)
    return aindex

def getRandomTextFromIndex(aIndex):
    res = -1
    while res != aIndex:
        aNumber = random.randint(0, len(y_train) - 1)
        res = y_train[aNumber]
    return x_train[aNumber]

print("ready")
s = " "
while s:
    category = classify(s)
    print("Movie",category)
    text = getRandomTextFromIndex(category)
    print("Chatbot:" + text)
    s = input("Human:")