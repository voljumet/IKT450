import numpy as np
import training_func as tf
import torch
import random as rand
from time import time


def stopwatch(ok):
	start = time()
	result = ok
	stop = time()
	print("Testing time: ", (stop - start))
	return result


def test(x_test, y_test, max_words, unique_words):
	
	inverted_y_set = tf.convert_array_shortanswers(y_test)      #this list will be inverted !!!!!!!!!!!

	result = []
	for each_question in x_test:
		result.append(classify(each_question, max_words, unique_words))

	counter = 0
	for k in range(len(x_test)):
		if result[k] != inverted_y_set[k]:
			counter += 1

	print(" Testing - Accuracy:", counter/len(y_test))


def classify(model, question, max_words, unique_words):
	indices = tf.make_text_into_numbers(question, max_words, unique_words)

	if torch.cuda.is_available():
		tensor = torch.LongTensor([indices]).cuda()
	else:
		tensor = torch.LongTensor([indices])

	y_pred = model(tensor).cpu().detach().numpy()
	answer = np.argmax(y_pred)
	return answer


def getRandomTextFromIndex(aIndex, y_train, x_train):
	res = -1
	while res != aIndex:
		aNumber = rand.randint(0, len(y_train) - 1)
		res = y_train[aNumber]
	return x_train[aNumber]

#
# predictions_test = []
# 			for eaaa in y_pred_train:
# 				newa = np.zeros(2)
# 				if eaaa[0].tolist() > eaaa[1].tolist():
# 					newa[0] = 1.0
# 				else:
# 					newa[1] = 1.0
# 				predictions_test.append(newa)
#
# 			accuracy_test = y_train.tolist()
#
# 			counter = 0
# 			for k in range(len(predictions_test)):
# 				if predictions_test[k].tolist() == accuracy_test[k]:
# 					counter += 1
#
# 			acc_test = counter/len(predictions_test)
#
# 			if (i % 250) == 0:
# 				print("Epoch:", i, "\nloss:", loss_train.cpu().detach().numpy(), "Accuracy:", acc_test)