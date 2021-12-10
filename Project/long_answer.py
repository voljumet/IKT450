import torch
import random
import training_func as tf
import testing_func as tes
from datasets import list_datasets, load_dataset, list_metrics, load_metric, get_dataset_config_names, \
    get_dataset_split_names
from sklearn.model_selection import train_test_split

''' Code running on a machine with enough diskspace available? requires ~120GB '''
local = True

if local:
    dataset = tf.json_reader('training-data/mydata*.json')
else:
    dataset = load_dataset('natural_questions', split='train')
''' -------------------------------------------------------------------------- '''
categories = ["environment", "politics", "advertisement", "public health", "research", "science", "music",
              "elections", "economics", "sport", "education", "business", "technology", "history", "entertainment"]

# load training data set
questions, short_answers, long_answer, labels = tf.load_data(dataset, local)

x_train1, x_test1, y_train1, y_test1 = train_test_split(questions, labels, test_size=0.2, random_state=42, shuffle=True)

# contain long answers
x_train_org = []
for i in range(len(long_answer)):
    x_train_org.append(' '.join(long_answer[i]))

# contains labels for x_train
y_train_org = []
for n in range(len(x_train1)):
    y_train_org.append(categories.index(y_train1[n]))

# fix short answer array
short_answers = tf.short_answers_make(short_answers)

print("Natural Language Process started ...")
x_train_temp, all_words = tf.split_dataset(x_train1)
print("Done!")


num_categories = len(categories)

# contains labels for x_train
y_train = []
for n in [categories.index(i) for i in y_train1]:
    y_train.append([0 for i in range(num_categories)])
    y_train[-1][n] = 1
    pass

unique_words = list(set(all_words))

# contains x_train converted to numbers
x_train = []

# take "max_words" amount of words and put it in an array as a number pointing to the words index in the "uniquewords" array
max_words = 11

for each in x_train_temp:
    x_train.append(tf.makeTextIntoNumbers1(each, max_words, unique_words))

# convert x_train y_train to tensors
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

n_steps = 10000


############################ Test data set ##########################
#test_dataset = tf.json_reader('validation-data/mydata*.json')
#load test data set
#test_questions, test_short_answers, test_long_answer, test_labels = tf.load_data(test_dataset, local)

#x_test_org = []
#for i in range(len(test_long_answer)):
#    x_test_org.append(' '.join(test_long_answer[i]))

y_test_org = []
for n in range(len(x_test1)):
    y_test_org.append(categories.index(y_test1[n]))
# fix short answer array
#test_short_answers = tf.short_answers_make(test_short_answers)

print("Test Natural Language Process started ...")
x_test_temp, test_all_words = tf.split_dataset(x_test1)
print("Done!")

# contains labels for x_train
y_test = []

for n in [categories.index(i) for i in y_test1]:
    y_test.append([0 for i in range(num_categories)])
    y_test[-1][n] = 1
    pass

test_nique_words = list(set(test_all_words))

# contains x_train converted to numbers
x_test = []
max_words = 11
for each in x_test_temp:
    x_test.append(tf.makeTextIntoNumbers1(each, max_words, test_nique_words))

# convert x_train y_train to tensors
if torch.cuda.is_available():
    print("Running on CUDA ...")
    using_cuda = True
    x_test = torch.LongTensor(x_test).cuda()
    y_test = torch.Tensor(y_test).cuda()
else:
    print("Running on CPU ...")
    using_cuda = False
    x_test = torch.LongTensor(x_test)
    y_test = torch.Tensor(y_test)
####################################################################
''' --------------------- TRAIN --------------------- '''
# True = load trained model from file
# False = train the model then save as file
file_name = f"trained_steps_{n_steps}_maxwords_{max_words}_datasize_{len(x_train)}_V1.pth"
nene =  tf.training_from_file(use_model=False, n_steps=n_steps, x_train=x_train, y_train=y_train, file_name=file_name,
                      unique_words=unique_words, questions= questions, max_words= max_words,  y_test= y_test,  x_test=x_test, y_train_org = y_train_org, y_test_org = y_test_org)
''' --------------------- TRAIN ---------------------'''

def getRandomTextFromIndex(aIndex):
    res = -1
    while res != aIndex:
        aNumber = random.randint(0, len(y_train_org) - 1)
        res = y_train_org[aNumber]
    return long_answer[aNumber]


print("ready")
s = " "
print("Length of unique_words: ", len(unique_words))
j = 0
count = 0
for i in questions:
    category = tes.classify(nene, i, max_words, unique_words)
    if category == y_train_org[j]:
        count += 1
    j+= 1
print("count: ", count)
print("Question: ", len(questions))
print("Accuracy: ", count/len(questions))
# while s:
#     category = tes.classify(nene ,s, max_words, unique_words)
#     print("category: ", category)
#
#     text = getRandomTextFromIndex(category)
#     print("Chatbot:" + ' '.join(text))
#     s = input("Human:")
