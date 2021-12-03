import torch

import training_func as tf
import testing_func as tes

from datasets import list_datasets, load_dataset, list_metrics, load_metric, get_dataset_config_names, \
	get_dataset_split_names


''' Code running on a machine with enough diskspace available? requires ~120GB '''
local = True

if local:
	dataset = tf.json_reader('Data2/mydata*.json')
else:
	dataset = load_dataset('natural_questions', split='train')
''' -------------------------------------------------------------------------- '''

# load data set
questions, short_answers, long_answer, labels = tf.load_data(dataset, local)

# fix short answer array
short_answers = tf.short_answers_make(short_answers)


print("Natural Language Process started ...")
x_train_temp, all_words = tf.split_dataset(questions)
print("Done!")

categories = list(set(short_answers))
y_train = []
for n in [categories.index(i) for i in short_answers]:
	y_train.append([0 for i in range(len(categories))])
	y_train[-1][n] = 1

unique_words = list(set(all_words))
x_train = []

# take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
max_words = 11

for each in x_train_temp:
	x_train.append(tf.makeTextIntoNumbers(each, max_words, unique_words))

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



train_loss = []
validate_loss = []

train_loss_acc = []
validate_acc = []

n_steps = 3000

''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
file_name = f"trained_steps_{n_steps}_maxwords_{max_words}_datasize_{len(x_train)}_V1.pth"
tf.training_from_file(use_model=False, n_steps=n_steps, x_train=x_train, y_train=y_train, file_name=file_name, unique_words=unique_words)
''' --------------------- TRAIN ---------------------'''



print("ready")
text = "first question"
while text:
	# out, ewer = tf.nat_lang_proc(text)
	category = tes.classify(text, max_words, unique_words)

	if category == 0:
		answer = "No"
	else:
		answer = "Yes"

	print("category prdiction: ", answer)
	# text = getRandomTextFromIndex(category)
	# print("Chatbot:" + text)
	s = input("Ask question: ")