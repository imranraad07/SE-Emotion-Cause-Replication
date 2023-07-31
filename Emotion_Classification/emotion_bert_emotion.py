import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

import re
from transformers import get_linear_schedule_with_warmup

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


import sys
import string
import re

epochs = 50
delta = 0.01

col = sys.argv[1]
flag = int(sys.argv[2])

print(col)

if flag == 0:
    model_name = "bert"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )



elif flag == 1:
    model_name = "roberta"


    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )



model.to(device)

# Load the training and validation data from CSV files
train_df = pd.read_csv("datasets/SOEmotion-Train.csv")
val_df = pd.read_csv("datasets/SOEmotion-Test.csv")


# df = pd.read_csv("datasets/JIRAEmotion.csv")
# comment_train_text = col + '_train_comment'
# label_train_text = col + '_train_label'
# comment_test_text = col + '_test_comment'
# label_test_text = col + '_test_label'

# train_df = df[[comment_train_text, label_train_text]]
# val_df = df[[comment_test_text, label_test_text]]
# Drop rows with null values
# train_df = train_df.dropna()
# val_df = val_df.dropna()

new_column_names = ["Text", col]

train_df.columns = new_column_names
val_df.columns = new_column_names



print(train_df.keys())


def text_cleaning(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    return text



# Define the dataset class
class CSVDataset(Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["Text"]
        text = text_cleaning(text)
        label = self.df.iloc[idx][col]

        # Tokenize the text and pad the sequences
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)

        # Convert the label to a tensor
        label = torch.tensor(label).long()

        return input_ids, label


# Set the maximum sequence length
MAX_LEN = 128

# Create the datasets
train_dataset = CSVDataset(train_df, tokenizer, MAX_LEN)
val_dataset = CSVDataset(val_df, tokenizer, MAX_LEN)

print(len(train_dataset))
print(len(val_dataset))

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


f1 = -0.1
loss_val = 100

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

f1 = -0.1
loss_val = 100


validation_loss_counter = 0
val_loss = 100


# Train the model
for epoch in range(epochs):
    total_train_loss = 0
    model.train()  # Put the model into training mode

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()        

        # Forward pass
        loss, logits = model(b_input_ids, labels=b_labels)[:2]

        # Accumulate the training loss
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update the learning rate
        scheduler.step()

    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_dataloader)            

    if avg_train_loss < loss_val:
        loss_val = avg_train_loss

    # ========================================
    #               Validation
    # ========================================
    model.eval()  # Put the model in evaluation mode

    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions , true_labels = [], []
    validation_loss = 0

    for batch in val_dataloader:
        b_input_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():        
            loss, logits = model(b_input_ids, labels=b_labels)[:2]
                
        total_eval_loss += loss.item()

        # Accumulate the training loss
        validation_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Calculate the average loss over all of the batches
    avg_test_loss = validation_loss / len(val_dataloader)

    # Flatten the predictions and true values for aggregate evaluation on all classes.
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    pred_flat = np.argmax(predictions, axis=1).flatten()

    # Calculate the validation accuracy of the model
    val_f1_score = f1_score(true_labels, pred_flat)

    print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_test_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")

    if avg_train_loss < delta:
        break

    if val_loss > avg_test_loss:
        val_loss = avg_test_loss 
        print(confusion_matrix(true_labels, pred_flat), val_f1_score)
        print(classification_report(true_labels, pred_flat))
        # my_array_pred = np.array(pred_flat)
        # my_array_true = np.array(true_labels)
        # df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
        # df.to_csv(model_name+'_pred_'+col+'_' + str(epoch)+ '_' +  str(val_f1_score) +'_.csv', index=False)

print(f1)
