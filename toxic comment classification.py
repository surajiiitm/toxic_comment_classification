import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")

train.isnull().any()

x_train = train.iloc[:, 1]
y_train = train.iloc[:, 2:8].values

import nltk
nltk.download("stopwords")
import re
import nltk
from nltk.corpus import stopwords
stopword = stopwords.words("english")

# remove unnecessary words
text = x_train[0]

def comment_to_words(comments):
    comments = str(comments)
    letters_only = re.sub('[^A-Za-z]', " ", comments)
    letters = letters_only.lower()

    words = letters.split()

    # removing stopword using nltk
    # remove stop words from words
    vector_words = [word for word in words if not word in stopword]
    return (" ".join(vector_words))

text = comment_to_words(text)

num_comment = len(x_train)
processed_comment = []
for i in range(0, num_comment):
    if((i+1)%1000 == 0 ):
        print("review %d of %d" %((i+1), num_comment))
    processed_comment.append(comment_to_words(x_train[i]))
    

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=2000)

train_data_features = vectorizer.fit_transform(processed_comment)
train_data_features = train_data_features.toarray()

print(train_data_features.shape)

