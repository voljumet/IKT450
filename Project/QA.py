import glob
import json
from random import random
import re
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rand

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from datasets import list_datasets, load_dataset, list_metrics, load_metric, get_dataset_config_names, \
	get_dataset_split_names


def short_answers_make(input):
	array = []
	for each in input:
		array.append(each[0])
	return array


def json_reader(path):
	dataset = []
	for file in glob.glob(path, recursive=True):
		with open(file) as json_file:
			dataset.append(json.load(json_file))
	return dataset


''' Code running on a machine with enough diskspace available? requires ~120GB '''
local = True

if local:
	dataset = json_reader('Data/mydata*.json')
else:
	dataset = load_dataset('natural_questions', split='train')
''' -------------------------------------------------------------------------- '''


def filter_html(data):
	long_answer_temp = []
	for i, each in enumerate(data['is_html']):
		if each == 0:
			long_answer_temp.append(data['token'][i])
	return long_answer_temp


def load_data(dataset):
	questions, short_answers, long_answer = [], [], []
	for each in dataset:
		if local:
			if ((each[1]) != [-1]):
				questions.append(each[0])
				short_answers.append(each[1])
				long_answer.append(filter_html(each[2]))
		else:
			if ((each["annotations"]["yes_no_answer"]) != [-1]):
				questions.append(each['question']['tokens'])
				short_answers.append(each['annotations']['yes_no_answer'])
				long_answer.append(filter_html(each['document']['tokens']))
	return questions, short_answers, long_answer


# def new(data):
#     save_data = []
#     save_data.append(data["question"]["tokens"])
#     save_data.append(data["annotations"]["yes_no_answer"])
#     save_data.append(data["document"]["tokens"])
#     return save_data
#
# def json_write(data):
#     for i, each in enumerate(data):
#         if((data[i]["annotations"]["yes_no_answer"]) != [-1]):
#             with open('Data/mydata'+str(i)+'.json', 'w') as f:
#                 json.dump(new(data[i]), f)
#
# print("Done filtering")


questions, short_answers, long_answer = load_data(dataset)

# fix short answer array
short_answers = short_answers_make(short_answers)

print("Data read!")
try:
	stopwords = set(stopwords.words('english'))
	lm = WordNetLemmatizer()
except:
	print("Did not find module stopwords or lemm, downloading ...")
	nltk.download('stopwords')
	nltk.download('wordnet')


def nat_lang_proc(all_questions):
	removed_stopwords = []
	all_words = []
	for each_question in all_questions:
		for each_word in each_question:
			# each_question[each_question.index(each_word)] = re.sub(r"[^a-zA-Z0-9]","", each_word) # DOES NOT WORK!!!
			if each_word.lower() in stopwords:
				each_question.remove(each_word)
			else:
				each_question[each_question.index(each_word)] = lm.lemmatize(each_word.lower())
				all_words.append(each_word.lower())
		removed_stopwords.append(each_question)
	return removed_stopwords, all_words


print("Removing stopwords ...")
x_train_temp, all_words = nat_lang_proc(questions)
print("Stopwords removed ...")

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

if torch.cuda.is_available():
	print("Running on CUDA ...")
	using_cuda = True
	x_train = torch.LongTensor(x_train).cuda()
	y_train = torch.Tensor(y_train).cuda()
else:
	print("Running on CPU ...")
	using_cuda = False
	x_train = torch.LongTensor(x_train)
	y_train = torch.Tensor(y_train)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		if using_cuda:
			self.embedding = nn.Embedding(len(unique_words), 20).cuda()

			self.lstm = nn.LSTM(input_size=20, hidden_size=16, num_layers=1,
			                    batch_first=True, bidirectional=False).cuda()

			self.fc1 = nn.Linear(16, 256).cuda()
			self.fc2 = nn.Linear(256, 2).cuda()
			self.sigmoid = nn.Sigmoid().cuda()
		else:
			self.embedding = nn.Embedding(len(unique_words), 20)
			self.lstm = nn.LSTM(input_size=20, hidden_size=16, num_layers=1,
			                    batch_first=True, bidirectional=False)

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
	return sum(l) / len(l)


n_steps = 3000


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
				print("loss:", loss_train.cpu().detach().numpy(), "- step:", i)

		torch.save(nene.state_dict(), f"trained_steps:{n_steps}_maxwords:{max_words}_datasize:{len(x_train)}_V1.pth")
		print("Training complete! ...")


''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
training_from_file(False)

''' --------------------- TRAIN ---------------------'''


def classify(line):
	indices = makeTextIntoNumbers(line)
	if using_cuda:
		tensor = torch.LongTensor([indices]).cuda()
	else:
		tensor = torch.LongTensor([indices])

	output = nene(tensor).cpu().detach().numpy()
	aindex = np.argmax(output)
	return aindex


def getRandomTextFromIndex(aIndex):
	res = -1
	while res != aIndex:
		aNumber = rand.randint(0, len(y_train) - 1)
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