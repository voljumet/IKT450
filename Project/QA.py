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
from datasets import list_datasets, load_dataset, list_metrics, load_metric, get_dataset_config_names, get_dataset_split_names
# check_if_cuda()

''' dev function to simply load from local file '''

def short_answers_make(input):
    array = []
    for each in input:
        array.append(each[0])
    return array

def json_reader(path):
    for file in glob.glob(path, recursive=True):
        with open(file) as json_file:
            dataset.append(json.load(json_file))


local = True

if local:
    dataset = []

    json_reader('Data/mydata*.json')
else:
    dataset = load_dataset('natural_questions', split='train')

''' --------------------------------'''
questions = []
short_answers = []
long_answer = []

def filter_html(data):
    long_answer_temp = []
    for i, each in enumerate(data['is_html']):
        if each == 0:
            long_answer_temp.append(data['token'][i])
    return long_answer_temp

for each in dataset:
    if local:
        questions.append(each[0])
        short_answers.append(each[1])
        long_answer.append(filter_html(each[2]))
    else:
        questions.append(each['question']['tokens'])
        short_answers.append(each['annotations']['yes_no_answer'])
        long_answer.append(filter_html(each['document']['tokens']))

# fix short answer array
short_answers = short_answers_make(short_answers)


print("Data read!")
stopwords = stopwords.words('english')


def remove_stopwords(all_questions):
    removed_stopwords = []
    all_words = []
    for each_question in all_questions:
        for each_word in each_question:
            if each_word.lower() in stopwords:
                each_question.remove(each_word)

            all_words.append(each_word.lower())
        removed_stopwords.append(each_question)
    return removed_stopwords, all_words



x_train_temp, all_words = remove_stopwords(questions)
categories = list(set(short_answers))
y_train = []
for n in [categories.index(i) for i in short_answers]:
    y_train.append([0 for i in range(len(categories))])
    y_train[-1][n] = 1

unique_words = list(set(all_words))
x_train = []

# take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
max_words = 11

def makeTextIntoNumbers(text):
    numbers = np.zeros(max_words, dtype=int)
    for count, each_word in enumerate(text):
        if count == max_words:
            break
        try:
            numbers[count] = unique_words.index(each_word)
        except ValueError:
            continue

    return list(numbers[:max_words])

for each in x_train_temp:
    x_train.append(makeTextIntoNumbers(each))


x_train = torch.LongTensor(x_train)
y_train = torch.Tensor(y_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(len(unique_words), 20)

        self.lstm = nn.LSTM(input_size=20,
                            hidden_size=16,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e = self.embedding(x)
        output, hidden = self.lstm(e)

        X = self.fc1(output[:, -1, :])
        X = F.relu(X)

        X = self.fc2(X)
        X = torch.sigmoid(X)

        return X


batch_size = 1
epochs = 5

nene = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(nene.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

train_loss = []
validate_loss = []

train_loss_acc = []
validate_acc = []

def avg(l):
    return sum(l)/len(l)

n_steps = 30000

def training_from_file(bool):
    if bool:
        nene.load_state_dict(torch.load(f"trained_steps:{n_steps}_maxwords:{max_words}_datasize:{len(x_train)}_V1.pth"))
    else:
        for i in range(n_steps):
            y_pred_train = nene(x_train)
            loss_train = loss_fn(y_pred_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if (i % 250) == 0:
                print("loss:", loss_train.detach().numpy(), "- step:", i)

        torch.save(nene.state_dict(), "trained_n.pth")


''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
training_from_file(False)

''' --------------------- TRAIN ---------------------'''


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
    print("Movie", category)
    text = getRandomTextFromIndex(category)
    print("Chatbot:" + text)
    s = input("Human:")