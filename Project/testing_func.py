import numpy as np
import training_func as tf
import torch
import random as rand


def classify(model ,line, max_words, unique_words):
	indices = tf.makeTextIntoNumbers1(line, max_words, unique_words)
	if torch.cuda.is_available():
		tensor = torch.LongTensor([indices]).cuda()
	else:
		tensor = torch.LongTensor([indices])
	output = model(tensor).cpu().detach().numpy()
	aindex = np.argmax(output)
	return aindex


def getRandomTextFromIndex(aIndex, y_train, x_train):
	res = -1
	while res != aIndex:
		aNumber = rand.randint(0, len(y_train) - 1)
		res = y_train[aNumber]
	return x_train[aNumber]