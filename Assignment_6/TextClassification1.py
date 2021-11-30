import pickle
import enchant
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import random
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import random
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords

import csv
import urllib
import sys

numpy.random.seed(7)


def remove_html(html):
    # parse html content
    soup = bs(html, "html.parser")
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


# Read data and pick number of rows and columns used
rows = 73000
dataset_tag = pd.read_csv("archive/Tags.csv", nrows=rows)
dataset_questions = pd.read_csv("archive/Questions.csv", nrows=rows, encoding="ISO-8859-1", usecols=[0, 5])
dataset_answers = pd.read_csv("archive/Answers.csv", nrows=rows, usecols=[3, 4, 5])
print("Data read!")

# remove "python" from all tags, since all of them have it.
dataset_tag = dataset_tag[dataset_tag.Tag != "python"]


# Remove the low score answers
def remove_low_score(dataset):
    final_dataset = pd.DataFrame({'ParentId': [], 'Score': [], 'Body': []})
    settet = set([])
    for id in dataset['ParentId']:
        settet.add(id)

    for each_index in settet:
        temp = pd.DataFrame({'ParentId': [], 'Score': [], 'Body': []})
        index_list = dataset.index[dataset['ParentId'] == each_index]
        for each in index_list:
            temp = temp.append(dataset.loc[[each]])
        final_dataset = final_dataset.append(temp.loc[temp['Score'] == temp['Score'].max(axis=0)])
        del final_dataset['Score']
    return final_dataset


# keep only the highest voted answers
dataset_answers = remove_low_score(dataset_answers)

# Using beautifulSoup to remove html encoding
dataset_questions["Body"] = dataset_questions.apply(lambda x: remove_html(x.Body), axis=1)
dataset_answers["Body"] = dataset_answers.apply(lambda x: remove_html(x.Body), axis=1)

# make a copy of the sets to use for training without stopwords
dataset_questions_training = dataset_questions.copy()
dataset_answers_training = dataset_answers.copy()

stop = stopwords.words('english')


def remove_stopwords(dataset):
    ''' BYTT SPLIT MED TOKENIZE ----------------------------------------------------------XXXXXXXXXXXXXXXXXXXXXXXX '''
    return dataset['Body'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop]))


# Removing stopwords from Answers and Questions
dataset_questions_training['Body'] = remove_stopwords(dataset_questions_training)
dataset_answers_training['Body'] = remove_stopwords(dataset_answers_training)


# Merging Tags and Questions for training
dataset_questions_tags_train = pd.merge(dataset_tag, dataset_questions_training, on='Id')

# split the dataset into body and tag
x_train_temp_question = dataset_questions_tags_train['Body']
y_train_temp_question = dataset_questions_tags_train['Tag']


# Merging Tags and Answers
dataset_answers_train = pd.merge(dataset_tag, dataset_answers.set_index('ParentId'), left_on='Id', right_index=True)

# split the dataset into answers and tags
x_train_temp_answer = dataset_answers_train['Body']
y_train_temp_answer = dataset_answers_train['Tag']


''' Task 2 needs qusetions and answers merged to generate something '''
#Merging Questions and Answers
# dataset_answers_and_questions_train = pd.merge(dataset_questions, dataset_answers.set_index('ParentId'), left_on='Id', right_index=True)
# x_train_temp_answer = dataset_answers_train['Body']
# y_train_temp_answer = dataset_answers_train['Tag']

#import pdb;pdb.set_trace()

# creating set of categories
categories = list(set(y_train_temp_question))

print("Number of categories: {}".format(len(categories)))

# print(categories.index('350'))


# creates an array that holds the category number from the "categories" array, instead of the string
y_train_org = np.array([categories.index(i) for i in y_train_temp_question])
y_train_org_ans = np.array([categories.index(i) for i in y_train_temp_answer])

# copy the data array
x_train_org = x_train_temp_question[:]
x_train_org_ans = x_train_temp_answer[:]


# count how many categories there are
num_categories = len(categories)

# create a label for each question as an array (ex: [ [1,0,0,0], [0,1,0,0],  [0,0,1,0],  [0,0,0,1] ] )
y_train = []
for n in [categories.index(i) for i in y_train_temp_question]:
    y_train.append([0 for i in range(num_categories)])
    y_train[-1][n] = 1
    pass


#import pdb;pdb.set_trace()

#Embeddings
allwords = ' '.join(x_train_temp_question).lower().split(' ')
uniquewords = list(set(allwords))

#Stemming
# from nltk.stem import *
# stemmer = PorterStemmer()

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
            numbers[count] = uniquewords.index(each_word)
        except ValueError:
            continue

    return list(numbers[:max_words])

# fills the x_train array with each question as an array or numbers
for question in x_train_temp_question:
    t = makeTextIntoNumbers(question)
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
        # Embedding(( ), 8-1024 dimensions)
        self.embedding = nn.Embedding(len(uniquewords), 128)
        
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_categories)

    def forward(self, inp):
        e = self.embedding(inp)
        output, hidden = self.lstm(e)

        x = self.fc1(output[:, -1, :])
        x = F.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

n = Net2()
print("Model", n)
print("Parameters", [param.nelement() for param in n.parameters()])

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(n.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()


def avg(listen):
    return sum(listen)/len(listen)


y_pred_train = 0
loss_train = 0
n_steps = 10000


def training_from_file(bool):
    if bool:
        n.load_state_dict(torch.load("trained_n.pth"))
    else:
        for i in range(n_steps):
            y_pred_train = n(x_train)
            loss_train = loss_fn(y_pred_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if (i % 25) == 0:
                print("loss:", loss_train.detach().numpy(), "- step:", i)

        torch.save(n.state_dict(), "trained_n.pth")


''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
training_from_file(False)

''' --------------------- TRAIN ---------------------'''


def classify(line):
    indices = makeTextIntoNumbers(line)
    tensor = torch.LongTensor([indices])
    output = n(tensor).detach().numpy()
    aindex = np.argmax(output)
    return aindex


def getRandomTextFromIndex(aIndex):
    res = -1
    while res != aIndex:
        a_number = random.randint(0, len(y_train_org_ans) - 1)
        res = y_train_org_ans[a_number]
    return x_train_org_ans[a_number]


def clean_input(input_string):
    string_input = pd.DataFrame({'Body': [input_string]})
    return remove_stopwords(string_input)


print("ready")

s = "Frist osx array"
while s:
    s = clean_input(s)
    print(s)
    category_index_num = classify(s[0])
    # print("Category: ", category)
    print("Category len: ", len(categories))
    print("Category index num: ", category_index_num)
    print("Category: ", categories[int(category_index_num)])
    text = getRandomTextFromIndex(category_index_num)
    print("Chatbot:" + text)
    # trenger å remove stop words fra input
    s = input("Human:")

for line in x_train_temp_question[10:]:
    classify(line)
