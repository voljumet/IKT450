import pickle
import json
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

def remove_html(html):
    # parse html content
    soup = bs(html, "html.parser")
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)
# test = get_dataset_split_names('natural_questions')
# YesNoAnswer = dataset[0]['annotations']['yes_no_answer']
# Question = dataset[0]['question']
# AnswerasTokens = dataset[0]['document']['tokens']['token']


''' dev function to simply load from local file '''

def oppe():
    with open('file.json') as json_file:
        return json.load(json_file)

dataset = oppe()

''' --------------------------------'''
questions = []
answers = []
answer_long_temp = []
answer_long = []


questions.append(dataset['question']['tokens'])
answers.append(dataset['annotations']['yes_no_answer'])

answer_long_temp.append(dataset['document']['tokens'])

# remove all but short answer data


# uses 'is_html' to filer out html from the long answer
for i, each in enumerate(answer_long_temp[0]['is_html']):
    if each == 0:
        answer_long.append(answer_long_temp[0]['token'][i])


print("Data read!")

# remove all but yes_no_answer
# for each in dataset_answers:
#   if yes_no_answer != -1

stopwords = stopwords.words('english')
def remove_stopwords(dataset):
    return dataset['Body'].apply(lambda stop_word: ' '.join([word for word in stop_word.lower() if word not in stopwords]))


# run remove_stopwords on data
questions = remove_stopwords(questions)

categories = list(set(y_train_temp_question))




