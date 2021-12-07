import glob
import json
import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split


try:
	stopwords = set(stopwords.words('english'))
	lm = WordNetLemmatizer()
except:
	print("Did not find module stopwords or lemm, downloading ...")
	nltk.download('stopwords')
	nltk.download('wordnet')


def convert_array_shortanswers(input):
	# This function converts from [[0,1],[1,0],[1,0],[..]..] to [0,1,1,..]
	array = []
	for each in input:
		array.append(each[0])
	return array


def jsonfile_reader(path):
	dataset = []
	for file in glob.glob(path, recursive=True):
		with open(file) as json_file:
			dataset.append(json.load(json_file))
	return dataset


def remove_html(data):
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
				long_answer.append(remove_html(each[2]))
		else:
			if ((each["annotations"]["yes_no_answer"]) != [-1]):
				questions.append(each['question']['tokens'])
				short_answers.append(each['annotations']['yes_no_answer'])
				long_answer.append(remove_html(each['document']['tokens']))
	return questions, short_answers, long_answer


def natural_lang_process_one_question(question):
	for each_word in question:
		# each_question[each_question.index(each_word)] = re.sub(r"[^a-zA-Z0-9]","", each_word) # DOES NOT WORK!!!
		if each_word.lower() in stopwords:
			question.remove(each_word)
		else:
			question[question.index(each_word)] = lm.lemmatize(each_word.lower())
	return question


def natural_lang_process_all_questions(all_questions):
	sentences = []
	all_words = []
	for each_question in all_questions:
		sent = natural_lang_process_one_question(each_question)
		all_words += sent
		sentences.append(sent)
	return sentences, all_words


def make_text_into_numbers(text, max_words, unique_words):
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
	def __init__(self, len_unique_words):
		super(Net, self).__init__()

		if torch.cuda.is_available():
			self.embedding = nn.Embedding(len_unique_words, 20).cuda()

			self.lstm = nn.LSTM(input_size=20, hidden_size=16, num_layers=1,
								batch_first=True, bidirectional=False).cuda()

			self.fc1 = nn.Linear(16, 256).cuda()
			self.fc2 = nn.Linear(256, 2).cuda()
			self.sigmoid = nn.Sigmoid().cuda()
		else:
			self.embedding = nn.Embedding(len_unique_words, 20)
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


def test_accuracy(label, prediction):
	counter = 0
	for k in range(len(label)):
		if label[k] == prediction[k]:
			counter += 1

	return counter / len(label)


def classify(model, indices, unique_words):
	if torch.cuda.is_available():
		tensor = torch.LongTensor([indices]).cuda()
	else:
		tensor = torch.LongTensor([indices])

	output = model(tensor).cpu().detach().numpy()
	aindex = np.argmax(output)
	return aindex


def printer(t_loss, t_acc, v_loss, v_acc):
	# Plot training & validation accuracy values
	fig = plt.figure(figsize=(10, 5))
	plt.plot(t_loss)
	plt.plot(v_loss)
	title1 = 'Model loss'+ '('+ str(datetime.datetime.today())+')'
	title2 = 'Model accuracy'+ '('+ str(datetime.datetime.today())+')'
	plt.title(title1)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Test', 'Train'], loc='upper left')
	fig.savefig(title1+'.jpg', bbox_inches='tight', dpi=150)
	plt.show()

	fig = plt.figure(figsize=(10, 5))
	plt.plot(t_acc)
	plt.plot(v_acc)
	plt.title(title2)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Test', 'Train'], loc='upper left')
	fig.savefig(title2+'.jpg', bbox_inches='tight', dpi=150)
	plt.show()


def using_cuda(x_train, y_train, x_test, y_test):
	if torch.cuda.is_available():
		print("Running on CUDA ...")
		x_train, x_test = torch.LongTensor(x_train).cuda(), torch.LongTensor(x_test).cuda()
		y_train, y_test = torch.Tensor(y_train).cuda(), torch.Tensor(y_test).cuda()
	else:
		print("Running on CPU ...")
		x_train, x_test = torch.LongTensor(x_train), torch.LongTensor(x_test)
		y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
	return x_train, y_train, x_test, y_test


def convert_tensor_to_list(inne):
	lista = []
	for each in inne.tolist():
		if each[0] > each[1]:
			lista.append(0)
		else:
			lista.append(1)
	return lista


def avg(l):
	return sum(l)/len(l)


def training_from_file(use_model, n_steps, x_temp, y_temp, file_name, len_unique_words):
	if use_model:
		torch.load(x_temp, "x_temp_tensor_" + file_name)
		torch.load(x_temp, "y_temp_tensor_" + file_name)
	else:
		torch.save(x_temp, "x_temp_tensor_" + file_name)
		torch.save(x_temp, "y_temp_tensor_" + file_name)

	x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42, shuffle=True)

	x_train, y_train, x_test, y_test = using_cuda(x_train, y_train, x_test, y_test)

	model = Net(len_unique_words)

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	# criterion = nn.CrossEntropyLoss()
	loss_fn = torch.nn.MSELoss()

	if use_model:
		model.load_state_dict(torch.load(file_name, map_location='cpu'))
		print("Trained model loaded from file, using the file: ", file_name)
	else:
		train_loss, validate_loss = [],  []
		train_acc, validate_acc = [], []

		for i in range(n_steps):
			y_pred_train = model(x_train)
			loss_train = loss_fn(y_pred_train, y_train)

			optimizer.zero_grad()
			loss_train.backward()
			optimizer.step()

			y_pred_test = model(x_test)
			loss_test = loss_fn(y_pred_test, y_test)

			train_loss.append(loss_train.item())
			validate_loss.append(loss_test.item())

			acc_test = test_accuracy(convert_tensor_to_list(y_pred_test), convert_tensor_to_list(y_test))
			acc_train = test_accuracy(convert_tensor_to_list(y_pred_train), convert_tensor_to_list(y_train))

			if (i % 25 == 0):
				print(i, "acc test:", round(acc_test, 4), "acc train:", round(acc_train, 4),
				"loss test:", round(loss_train.item(), 4), "loss train:", round(loss_test.item(), 4))

			train_acc.append(acc_test)
			validate_acc.append(acc_train)

		printer(train_loss, train_acc, validate_loss, validate_acc)

		torch.save(model.state_dict(), file_name)
		print("Model training complete!")
	return model



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
