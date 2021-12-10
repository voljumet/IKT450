import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

import training_func as tf
import testing_func as tes

# from datasets import load_dataset


train_from_file = False
local = True    # Code running on a machine with enough diskspace available? requires ~120GB
max_words = 12  # take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
n_steps = 5000

# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

if local:
	dataset = tf.jsonfile_reader('file.json')
# else:
	# dataset = load_dataset('natural_questions', split='test')


def remove_html(data):
	long_answer_temp = []
	for i, each in enumerate(data['is_html']):
		if each == 0:
			long_answer_temp.append(data['token'][i])
	return long_answer_temp


def zupp(start_token, tokens):
	temp = []
	for each in range(start_token - 3, start_token + 4):
		temp.append(tokens[each])

	return ' '.join(map(str, temp))

def create_data(dataset):
	total = []
	for each in dataset:
		tokens = each['document']['tokens']['token']
		for i in range(len(each['annotations']['short_answers'])):
			if each['annotations']['short_answers'][i]['start_token'][0]:
				start_byte = each['annotations']['short_answers'][i]['start_byte'][0]
				break

		search_string = zupp(start_byte, tokens)

		answer_text = each['annotations']['short_answers'][0]['text'][0]
		context = ' '.join(map(str, remove_html(each['document']['tokens'])))

		total.append({
			"context": context,
			"qas": [
				{
					"id": each['annotations']['id'][0],
					"is_impossible": False,
					"question": each['question']['text'],
					"answers": [
						{
							"text": answer_text,
							"answer_start": context.find(search_string) + search_string.find(answer_text),
						}
					],
				}
			],
		})
	return total


train = create_data(dataset)


# load data set
questions, yes_no_answer, long_answer, short_answer = tf.load_data(dataset, local)

# fix short answer array
yes_no_answer = tf.convert_array_shortanswers(yes_no_answer)


print("Natural Language Process started ...")
x_train_temp, all_words = tf.natural_lang_process_all_questions(questions)
print("Done!")

categories = list(set(yes_no_answer))
y_temp = []
for n in [categories.index(i) for i in yes_no_answer]:
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

num_lay = 1
input_s = 7
hidden = 5
out_in = 111
# for num_lay in range(1,2):
# 	for input_s in range(7,8):
# 		for hidden in range(5,6):
# 			for out_in in range(111,112):
model = tf.training_from_file(use_model=train_from_file, n_steps=n_steps, x_temp=x_temp, y_temp=y_temp, file_name=file_name, len_unique_words=len(unique_words), input_s=input_s, hidden=hidden, out_in=out_in, num_lay=num_lay)
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