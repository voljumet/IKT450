import torch

import training_func as tf
import testing_func as tes

from datasets import list_datasets, load_dataset, list_metrics, load_metric, get_dataset_config_names, \
	get_dataset_split_names
# si = ''
# while si != "T" or si != "P":
# 	si = input("Use pretrained file, or train new file? (P / T)")
# 	if si == "F":
# 		train_from_file = True
# 		break
# 	elif si == "T":
# 		train_from_file = False
# 		break
train_from_file = False
local = True    # Code running on a machine with enough diskspace available? requires ~120GB
max_words = 11  # take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
n_steps = 1000


if local:
	dataset = tf.jsonfile_reader('Test-Data/mydata*.json')
else:
	dataset = load_dataset('natural_questions', split='train')
	# dataset2 = load_dataset('natural_questions', split='test')

# load data set
questions, short_answers, long_answer = tf.load_data(dataset, local)

# fix short answer array
short_answers = tf.convert_array_shortanswers(short_answers)


print("Natural Language Process started ...")
x_train_temp, all_words = tf.natural_lang_process_all_questions(questions)
print("Done!")

categories = list(set(short_answers))
y_temp = []
for n in [categories.index(i) for i in short_answers]:
		y_temp.append([0 for i in range(len(categories))])
		y_temp[-1][n] = 1

unique_words = list(set(all_words))
x_temp = []


for each in x_train_temp:
	x_temp.append(tf.make_text_into_numbers(each, max_words, unique_words))


''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
file_name = f"trained_steps_{n_steps}_maxwords_{max_words}_datasize_{len(x_temp)}_V1.pth"
model = tf.training_from_file(use_model=train_from_file, n_steps=n_steps, x_temp=x_temp, y_temp=y_temp, file_name=file_name, unique_words=unique_words)
''' --------------------- TRAIN ---------------------'''

print("ready")
text = "first question"
while text:
	out = tf.natural_lang_process_one_question(text.split())
	category = tes.classify(model, text, max_words, unique_words)

	if category == 0:
		answer = "No"
	else:
		answer = "Yes"

	print("category prdiction: ", answer)
	# text = getRandomTextFromIndex(category)
	# print("Chatbot:" + text)
	text = input("Ask question: ")