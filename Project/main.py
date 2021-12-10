import training_func as tf
import testing_func as tes

from datasets import load_dataset

use_pretrained_model = False
local = True    # Code running on a machine with enough diskspace available? requires ~120GB
max_words = 12  # take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
n_steps = 5000


if local:
	dataset = tf.jsonfile_reader('Test-Data/mydata*.json')
else:
	dataset = load_dataset('natural_questions', split='train')
	# dataset2 = load_dataset('natural_questions', split='test')

# load data set
questions, yes_no_answer, text_tokens, short_answer = tf.load_data(dataset, local)

print("Natural Language Process started ...")
x_train_temp, unique_words = tf.natural_lang_process_all_questions(questions)
print("Done!")

categories = list(set(yes_no_answer))

y_temp = []
for n in [categories.index(i) for i in yes_no_answer]:
		y_temp.append([0 for i in range(len(categories))])
		y_temp[-1][n] = 1


x_temp = []
for each in x_train_temp:
	x_temp.append(tf.make_text_into_numbers(each, max_words, unique_words))


''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
num_lay = 1
input_s = 20
hidden = 16
out_in = 256

file_name = "_ins:" + str(input_s) + "_hid:" + str(hidden) + "_out:" + str(out_in) + "_lay:" + \
            str(num_lay) + f"trained_steps_{n_steps}_maxwords_{max_words}_datasize_{len(x_temp)}_V1.pth"

# for num_lay in range(1,2):
# 	for input_s in range(7,8):
# 		for hidden in range(5,6):
# 			for out_in in range(111,112):
model = tf.training_from_file(use_pretrained_model=use_pretrained_model, n_steps=n_steps, x_temp=x_temp, y_temp=y_temp, file_name=file_name, len_unique_words=len(unique_words), input_s=input_s, hidden=hidden, out_in=out_in, num_lay=num_lay)
''' --------------------- TRAIN ---------------------'''

print("ready")
text = "first question"
while text:
	out = tf.natural_lang_process_one_question(text.split())
	category = tes.classify(model, text, max_words)

	if category == 0:
		answer = "No"
	else:
		answer = "Yes"

	print("category prdiction: ", answer)
	# text = getRandomTextFromIndex(category)
	# print("Chatbot:" + text)
	text = input("Ask question: ")