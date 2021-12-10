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
	questions, yes_no_answer, short_answers, text_tokens = [], [], [], []

	for count, each in enumerate(dataset):
		if count == 100:
			break
		if local:
			if ((each[1]) != [-1]):
				questions.append(each[0])
				yes_no_answer.append(each[1])
				text_tokens.append(remove_html(each[2]))

		else:
			# if ((each["annotations"]["yes_no_answer"]) != [-1]):
			questions.append(each['question']['tokens'])
			yes_no_answer.append(each['annotations']['yes_no_answer'])
			text_tokens.append(remove_html(each['document']['tokens']))
			short_answers.append(each['annotations']['short_answers'][0]['text'][0])
	yes_no_answer = convert_array_shortanswers(yes_no_answer)
	return questions, yes_no_answer, text_tokens, short_answers


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
	return sentences, list(set(all_words))


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
	def __init__(self, len_unique_words, input_s, hidden, out_in, num_lay):
		super(Net, self).__init__()

		if torch.cuda.is_available():
			self.embedding = nn.Embedding(len_unique_words, input_s).cuda()

			self.lstm = nn.LSTM(input_size=input_s, hidden_size=hidden, num_layers=num_lay,
			                    batch_first=True, bidirectional=False).cuda()

			self.fc1 = nn.Linear(hidden, out_in).cuda()
			self.fc2 = nn.Linear(out_in, 2).cuda()
			self.sigmoid = nn.Sigmoid().cuda()
		else:
			self.embedding = nn.Embedding(len_unique_words, input_s)
			self.lstm = nn.RNN(input_size=input_s, hidden_size=hidden, num_layers=num_lay,
			                    batch_first=True, bidirectional=False)

			self.fc1 = nn.Linear(hidden, out_in)
			self.fc2 = nn.Linear(out_in, 2)
			self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		e = self.embedding(x)

		output, hidden = self.lstm(e)

		X = self.fc1(output[:, -1, :])
		X = F.relu(X)

		X = self.fc2(X)
		X = torch.sigmoid(X)

		return X


def test_fone(label, pred):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for k in range(len(label)):
		predd = pred[k]
		labbel = label[k]

		if labbel == 1 and predd == 1:
			TP += 1

		if labbel == 0 and predd == 0:
			TN += 1

		if labbel == 1 and predd == 0:
			FN += 1

		if labbel == 0 and predd == 1:
			FP += 1

	print("Accuracy:", (TP + TN) / (TP + TN + FP + FN))
	try:
		recall = TP / (TP + FN)
	except:
		recall = TP / (TP + FN + 0.00001)

	print("Recall", recall)
	print("Precision", TP / (TP + FP))
	print("F1", (2 * TP) / (2 * TP + FP + FN))


def test_accuracy(label, pred):
	counter = 0

	for k in range(len(label)):
		predd = pred[k]
		labbel = label[k]
		if labbel == predd:
			counter += 1

	return counter / len(label)


def classify(model, indices):
	if torch.cuda.is_available():
		tensor = torch.LongTensor([indices]).cuda()
	else:
		tensor = torch.LongTensor([indices])

	output = model(tensor).cpu().detach().numpy()
	aindex = np.argmax(output)
	return aindex


def printer(train_loss, train_acc, test_loss, test_acc):
	# Plot training & validation accuracy values
	fig = plt.figure(figsize=(5, 5))
	plt.plot(train_loss)
	plt.plot(test_loss)
	title1 = 'Model loss' + '(' + str(datetime.datetime.today()) + ')'
	plt.title(title1)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	fig.savefig("gr_loss/"+ title1 + '.jpg', bbox_inches='tight', dpi=150)
	plt.show()

	fig = plt.figure(figsize=(5, 5))
	plt.plot(train_acc)
	plt.plot(test_acc)
	title2 = 'Model accuracy' + '(' + str(datetime.datetime.today()) + ')'
	plt.title(title2)
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	fig.savefig("gr_acc/"+ title2 + '.jpg', bbox_inches='tight', dpi=150)
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
	return sum(l) / len(l)


def training_from_file(use_pretrained_model, n_steps, x_temp, y_temp, file_name, len_unique_words, input_s, hidden, out_in,
						num_lay):
	if use_pretrained_model:
		torch.load(x_temp, "x_temp_tensor_" + file_name)
		torch.load(y_temp, "y_temp_tensor_" + file_name)
	else:
		torch.save(x_temp, "x_temp_tensor_" + file_name)
		torch.save(y_temp, "y_temp_tensor_" + file_name)

	x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42, shuffle=True)
	x_train, y_train, x_test, y_test = using_cuda(x_train, y_train, x_test, y_test)

	model = Net(len_unique_words, input_s, hidden, out_in, num_lay)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_fn = torch.nn.MSELoss()
	# criterion = nn.CrossEntropyLoss()

	if use_pretrained_model:
		model.load_state_dict(torch.load("models/" + file_name, map_location='cpu'))
		print("Trained model loaded from file, using the file: ", file_name)
	else:
		train_loss, test_loss = [], []
		train_acc, test_acc = [], []

		for i in range(n_steps):
			# train loss
			y_pred_train = model(x_train)
			loss_train = loss_fn(y_pred_train, y_train)

			optimizer.zero_grad()
			loss_train.backward()
			optimizer.step()

			# test loss
			y_pred_test = model(x_test)
			loss_test = loss_fn(y_pred_test, y_test)

			# append train and test loss
			train_loss.append(loss_train.item())
			test_loss.append(loss_test.item())

			# compare train accuracy, compare test accuracy
			acc_train = test_accuracy(convert_tensor_to_list(y_pred_train), convert_tensor_to_list(y_train))
			acc_test = test_accuracy(convert_tensor_to_list(y_pred_test), convert_tensor_to_list(y_test))

			# append train loss, compare test loss
			train_acc.append(acc_train)
			test_acc.append(acc_test)

			if (i % 25 == 0):
				test_fone(convert_tensor_to_list(y_pred_test), convert_tensor_to_list(y_test))
				print(i, "acc test:", round(acc_test, 4), "acc train:", round(acc_train, 4),
				      "loss test:", round(loss_train.item(), 4), "loss train:", round(loss_test.item(), 4))

		# print loss and accuracy as graphs
		printer(train_loss, train_acc, test_loss, test_acc)
		torch.save(model.state_dict(), "models/" + file_name)
		print("Model training complete!")
	return model


''' Peshangs load function'''
# def new(data):
#     save_data = []
#     save_data.append(data["question"]["tokens"])
#     save_data.append(data["annotations"]["yes_no_answer"])
#     save_data.append(data["document"]["tokens"])
#	  save_data.append(data['annotations']['short_answers'][0]['text'][0])
#     return save_data
#
# def json_write(data):
#     for i, each in enumerate(data):
#         if((data[i]["annotations"]["yes_no_answer"]) != [-1]):
#             with open('Data/mydata'+str(i)+'.json', 'w') as f:
#                 json.dump(new(data[i]), f)
#
# print("Done filtering")
