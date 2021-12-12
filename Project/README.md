# Question-answering system, project for IKT450

The project is split into three parts:
1. yes-no answers
2. answers by classification
3. fine-tuning pre-trained BERT model
---
1
---
To test the first part, the file main.py can be used, but before running the file, 
a few parameters has to be set depending on how you want to run it:
        
    This variable is set to True if you want to train a new model
    use_pretrained_model = False

    # if this variable is set to False, it will download ≈120 GB of data. True will used the files in the folder.
    local = True    
    
    # max_words is the maximum number of words that will be used in each question.
    max_words = 12

    # number of epochs to train the model
    n_steps = 1000
---
2
---
To test the second part, the file long_answer.py on the branch "text_classification" can be used.
this will be moved to the main branch after the code is integrated.  



---
3
---
To test the last part, the file Question Answer Application.ipynb can be used.

This is only configured to run with the whole dataset that contains ≈120 GB of data.


---
the following text was the idea for the start of the project, and does not entirely reflect the current state of the project.

read the dataset  
create a dictionary:  
remove all but Yes_no_answer
remove HTML tags  
do some NLP using lemmitizer, stopwords, stemming, etc.  
returns y_train, y_validate, x_train, x_validate  

use y_train, x_train to train  
create functions for RNN using torchvision  
use y_validate, x_validate to validate accuracy. if accuracy is low keep training.  


use y_test, x_test to test model accuracy.  

take input from user and use NLP processing  


 V1 = NLP only removes stopwords, n_steps=30000, 669 questions, max_words=11, hidden_layers=256

    dataset = [['question','', ],[answer]]
    
    x_train_temp = dataset[0]
    y_train_temp = dataset[1]
    
    
    Get all "categories"
    giving Y_train[0] = {0 or 1}

    def preprocessing_data():
        Remove everything but yes_no_answer = yes/no
        Remove Stopwords, Stem, Lemmitize, etc.
        Get "all_words" from "X_train"
        Get all "unique_words" from "all_words"
        return X_train_temp (X_train_temp = [[0,1],[1,0],[0,1],[0,1],[0,1],[1,0],[1,0],[0,1],...])

    def make_text_into_numbers(X_train_temp):
        return X_train (X_train[0] = [word1_index, word2_index, word3_index, ..., word_10_index])

    x_train = torch.LongTensor(x_train)
    y_train = torch.Tensor(y_train)
    
    class Net(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            self.embedding = nn.Embedding(len(unique_words), 20)
    
            self.lstm = nn.LSTM(input_size=20,
                                hidden_size=10,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=False)
    
            self.fc1 = nn.Linear(10, 128)
            self.fc2 = nn.Linear(128, 2)
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            e = self.embedding(x)
            output, hidden = self.lstm(e)
    
            X = self.fc1(output[:, -1, :])
            X = F.relu(X)
    
            X = self.fc2(X)
            X = torch.sigmoid(X)
    
            return X
    
    model = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(n.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

