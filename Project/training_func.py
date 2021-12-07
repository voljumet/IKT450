import glob
import json
from time import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import testing_func as tes
import matplotlib.pyplot as plt
import datetime

from classification import classify_text as cf


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
	questions, short_answers, long_answer, label = [], [], [], []
	for each in dataset:
		if local:
			if ((each[1]) != [-1]):
				questions.append(each[0])
				short_answers.append(each[1])
				long_answer.append(filter_html(each[2]))
				start = time()
				result = cf(' '.join(each[0]))
				stop = time()
				print(stop-start)
				label.append(result['labels'][0])

		else:
			if ((each["annotations"]["yes_no_answer"]) != [-1]):
				questions.append(each['question']['tokens'])
				short_answers.append(each['annotations']['yes_no_answer'])
				long_answer.append(filter_html(each['document']['tokens']))
	return questions, short_answers, long_answer, label


try:
	stopwords = set(stopwords.words('english'))
	lm = WordNetLemmatizer()
except:
	print("Did not find module stopwords or lemm, downloading ...")
	nltk.download('stopwords')
	nltk.download('wordnet')


def nat_lang_proc(question):
	#for each_word in question:
		# each_question[each_question.index(each_word)] = re.sub(r"[^a-zA-Z0-9]","", each_word) # DOES NOT WORK!!!
		#if each_word.lower() in stopwords:
		#	question.remove(each_word)
		#else:
		#	question[question.index(each_word)] = lm.lemmatize(each_word.lower())
	return question


def split_dataset(all_questions):
	sentences = []
	all_words = []
	for each_question in all_questions:
		sent = nat_lang_proc(each_question)
		all_words += sent
		sentences.append(sent)
	return sentences, all_words

def makeTextIntoNumbers1(text, max_words, unique_words):
	numbers = np.zeros(max_words, dtype=int)
	for count, each_word in enumerate(text):
		if count == max_words:
			break
		try:
			numbers[count] = unique_words.index(each_word)
		except ValueError:
			continue

	return list(numbers[:max_words])
def makeTextIntoNumbers(text, max_words, unique_words):
	iwords = text.lower().split(' ')
	numbers = []
	for n in iwords:
		try:
			numbers.append(unique_words.index(n))
		except ValueError:
			numbers.append(0)
	numbers = numbers + [0,0,0,0,0]
	return numbers[:6]


class Net(nn.Module):
	def __init__(self, unique_words):
		super(Net, self).__init__()

		if torch.cuda.is_available():
			self.embedding = nn.Embedding(len(unique_words), 20).cuda()

			self.lstm = nn.LSTM(input_size=20, hidden_size=16, num_layers=1,
								batch_first=True, bidirectional=False).cuda()

			self.fc1 = nn.Linear(16, 128).cuda()
			self.fc2 = nn.Linear(128, 15).cuda()
			self.sigmoid = nn.Sigmoid().cuda()
		else:
			self.embedding = nn.Embedding(len(unique_words), 20)
			self.lstm = nn.LSTM(input_size=20, hidden_size=16, num_layers=1,
								batch_first=True, bidirectional=False)

			self.fc1 = nn.Linear(16, 128)
			self.fc2 = nn.Linear(128, 15)

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


def printer(t_acc, v_acc):
	# Plot training & validation accuracy values
	fig = plt.figure(figsize=(10, 5))
	title1 = 'Model loss'+ '('+ str(datetime.datetime.today())+')'
	title2 = 'Model accuracy'+ '('+ str(datetime.datetime.today())+')'
	plt.title(title1)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	fig.savefig(title1+'.jpg', bbox_inches='tight', dpi=150)
	plt.show()

	fig = plt.figure(figsize=(10, 5))
	plt.plot(t_acc)
	plt.plot(v_acc)
	plt.title(title2)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	fig.savefig(title2+'.jpg', bbox_inches='tight', dpi=150)
	plt.show()




def training_from_file(use_model, n_steps, x_train, y_train, file_name, unique_words, questions, max_words, y_train_org, y_test_org ,test_questions):
	train_acc = []
	validate_acc = []
	nene = Net(unique_words)

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(nene.parameters(), lr=0.001, weight_decay=0.0001)
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
				j = 0
				count = 0
				for q in questions:
					category = tes.classify(nene, q, max_words, unique_words)
					if category == y_train_org[j]:
						count += 1
					j += 1
				train_acc.append(count / len(questions))
				k = 0
				test_count = 0
				for t in test_questions:
					test_category = tes.classify(nene, t, max_words, unique_words)
					if test_category == y_test_org[k]:
						test_count += 1
					k += 1
				validate_acc.append(test_count / len(test_questions))
				print("loss:", loss_train.cpu().detach().numpy(), "- step:", i, " Training Accuracy: ", count / len(questions), " Testing Accuracy: ", test_count / len(test_questions))

		torch.save(nene.state_dict(), file_name)
		printer(train_acc, validate_acc)
		print("Model training complete!")
	return nene

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
