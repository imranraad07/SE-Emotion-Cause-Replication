import csv

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import argparse
import pandas as pd

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser(description="Read train and test data from CSV files")

# Add command-line arguments for train and test data filenames
parser.add_argument('--train_file', required=True, help="Path to the training data CSV file")
parser.add_argument('--test_file', required=True, help="Path to the testing data CSV file")
parser.add_argument('--emotion', required=True, help="Emotion")

# Parse the command-line arguments
args = parser.parse_args()

# Read the training and testing data from the provided filenames
train_data = pd.read_csv(args.train_file)
test_data = pd.read_csv(args.test_file)


train_file = args.train_file
test_file = args.test_file
emotion = args.emotion

import pandas as pd

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train = tfidf_vectorizer.fit_transform(train_data['Text'])
X_test = tfidf_vectorizer.transform(test_data['Text'])

svm_classifier = LinearSVC(C=1.0)
svm_classifier.fit(X_train, train_data[emotion])

predictions = svm_classifier.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(test_data[emotion], predictions)

# You can also print a more detailed classification report
print(classification_report(test_data[emotion], predictions))

