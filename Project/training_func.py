import glob
import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def filter_html(data):
	long_answer_temp = []
	for i, each in enumerate(data['is_html']):
		if each == 0:
			long_answer_temp.append(data['token'][i])
	return long_answer_temp


def load_data(dataset, local):
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


try:
	stopwords = set(stopwords.words('english'))
	lm = WordNetLemmatizer()
except:
	print("Did not find module stopwords or lemm, downloading ...")
	nltk.download('stopwords')
	nltk.download('wordnet')


def nat_lang_proc(question):
	for each_word in question:
		# each_question[each_question.index(each_word)] = re.sub(r"[^a-zA-Z0-9]","", each_word) # DOES NOT WORK!!!
		if each_word.lower() in stopwords:
			question.remove(each_word)
		else:
			question[question.index(each_word)] = lm.lemmatize(each_word.lower())
	return question


def split_dataset(all_questions):
	sentences = []
	all_words = []
	for each_question in all_questions:
		sent = nat_lang_proc(each_question)
		all_words += sent
		sentences.append(sent)
	return sentences, all_words



def makeTextIntoNumbers(text, max_words, unique_words):
	numbers = np.zeros(max_words, dtype=int)
	for count, each_word in enumerate(text):
		if count == max_words:
			break
		try:
			numbers[count] = unique_words.index(each_word)
		except ValueError:
			continue

	return list(numbers[:max_words])



class Net(nn.Module):
	def __init__(self, unique_words):
		super(Net, self).__init__()

		if torch.cuda.is_available():
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

def avg(number):
	return sum(number) / len(number)





def training_from_file(use_model, n_steps, x_train, y_train, file_name, unique_words):
	nene = Net(unique_words)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(nene.parameters(), lr=0.001)
	loss_fn = torch.nn.MSELoss()

	if use_model:
		nene.load_state_dict(torch.load(file_name, map_location='cpu'))
		print("Trained model loaded from file, using the file: ", file_name)
	else:
		for i in range(n_steps):
			y_pred_train = nene(x_train)
			loss_train = loss_fn(y_pred_train, y_train)

			optimizer.zero_grad()
			loss_train.backward()
			optimizer.step()
			if (i % 250) == 0:
				print("loss:", loss_train.cpu().detach().numpy(), "- step:", i)

		torch.save(nene.state_dict(), file_name)
		print("Model training complete!")


''' Peshangs load function'''
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
