# toxic_comment_classification
To build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.



```python
# import Libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
```


```python
train = pd.read_csv("train.csv")
```


```python
train.isnull().any()
```




    id               False
    comment_text     False
    toxic            False
    severe_toxic     False
    obscene          False
    threat           False
    insult           False
    identity_hate    False
    dtype: bool




```python
x_train = train.iloc[:, 1]
y_train = train.iloc[:, 2:8].values
```


```python
import nltk
nltk.download("stopwords")
import re
import nltk
from nltk.corpus import stopwords
stopword = stopwords.words("english")
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/surajiiitm/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.



```python
# remove unnecessary words
text = x_train[0]
text
```




    "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27"




```python
def comment_to_words(comments):
    comments = str(comments)
    letters_only = re.sub('[^A-Za-z]', " ", comments)
    letters = letters_only.lower()

    words = letters.split()

    # removing stopword using nltk
    # remove stop words from words
    vector_words = [word for word in words if not word in stopword]
    return (" ".join(vector_words))
```


```python
text = comment_to_words(text)
text
```




    'explanation edits made username hardcore metallica fan reverted vandalisms closure gas voted new york dolls fac please remove template talk page since retired'




```python
num_comment = len(x_train)
processed_comment = []
for i in range(0, num_comment):
    if((i+1)%1000 == 0 ):
        print("review %d of %d" %((i+1), num_comment))
    processed_comment.append(comment_to_words(x_train[i]))
```

    review 1000 of 159571
    review 2000 of 159571
    review 3000 of 159571
    review 4000 of 159571
    review 5000 of 159571
    review 6000 of 159571
    review 7000 of 159571
    review 8000 of 159571
    review 9000 of 159571
    review 10000 of 159571
    review 11000 of 159571
    review 12000 of 159571
    review 13000 of 159571
    review 14000 of 159571
    review 15000 of 159571
    review 16000 of 159571
    review 17000 of 159571
    review 18000 of 159571
    review 19000 of 159571
    review 20000 of 159571
    review 21000 of 159571
    review 22000 of 159571
    review 23000 of 159571
    review 24000 of 159571
    review 25000 of 159571
    review 26000 of 159571
    review 27000 of 159571
    review 28000 of 159571
    review 29000 of 159571
    review 30000 of 159571
    review 31000 of 159571
    review 32000 of 159571
    review 33000 of 159571
    review 34000 of 159571
    review 35000 of 159571
    review 36000 of 159571
    review 37000 of 159571
    review 38000 of 159571
    review 39000 of 159571
    review 40000 of 159571
    review 41000 of 159571
    review 42000 of 159571
    review 43000 of 159571
    review 44000 of 159571
    review 45000 of 159571
    review 46000 of 159571
    review 47000 of 159571
    review 48000 of 159571
    review 49000 of 159571
    review 50000 of 159571
    review 51000 of 159571
    review 52000 of 159571
    review 53000 of 159571
    review 54000 of 159571
    review 55000 of 159571
    review 56000 of 159571
    review 57000 of 159571
    review 58000 of 159571
    review 59000 of 159571
    review 60000 of 159571
    review 61000 of 159571
    review 62000 of 159571
    review 63000 of 159571
    review 64000 of 159571
    review 65000 of 159571
    review 66000 of 159571
    review 67000 of 159571
    review 68000 of 159571
    review 69000 of 159571
    review 70000 of 159571
    review 71000 of 159571
    review 72000 of 159571
    review 73000 of 159571
    review 74000 of 159571
    review 75000 of 159571
    review 76000 of 159571
    review 77000 of 159571
    review 78000 of 159571
    review 79000 of 159571
    review 80000 of 159571
    review 81000 of 159571
    review 82000 of 159571
    review 83000 of 159571
    review 84000 of 159571
    review 85000 of 159571
    review 86000 of 159571
    review 87000 of 159571
    review 88000 of 159571
    review 89000 of 159571
    review 90000 of 159571
    review 91000 of 159571
    review 92000 of 159571
    review 93000 of 159571
    review 94000 of 159571
    review 95000 of 159571
    review 96000 of 159571
    review 97000 of 159571
    review 98000 of 159571
    review 99000 of 159571
    review 100000 of 159571
    review 101000 of 159571
    review 102000 of 159571
    review 103000 of 159571
    review 104000 of 159571
    review 105000 of 159571
    review 106000 of 159571
    review 107000 of 159571
    review 108000 of 159571
    review 109000 of 159571
    review 110000 of 159571
    review 111000 of 159571
    review 112000 of 159571
    review 113000 of 159571
    review 114000 of 159571
    review 115000 of 159571
    review 116000 of 159571
    review 117000 of 159571
    review 118000 of 159571
    review 119000 of 159571
    review 120000 of 159571
    review 121000 of 159571
    review 122000 of 159571
    review 123000 of 159571
    review 124000 of 159571
    review 125000 of 159571
    review 126000 of 159571
    review 127000 of 159571
    review 128000 of 159571
    review 129000 of 159571
    review 130000 of 159571
    review 131000 of 159571
    review 132000 of 159571
    review 133000 of 159571
    review 134000 of 159571
    review 135000 of 159571
    review 136000 of 159571
    review 137000 of 159571
    review 138000 of 159571
    review 139000 of 159571
    review 140000 of 159571
    review 141000 of 159571
    review 142000 of 159571
    review 143000 of 159571
    review 144000 of 159571
    review 145000 of 159571
    review 146000 of 159571
    review 147000 of 159571
    review 148000 of 159571
    review 149000 of 159571
    review 150000 of 159571
    review 151000 of 159571
    review 152000 of 159571
    review 153000 of 159571
    review 154000 of 159571
    review 155000 of 159571
    review 156000 of 159571
    review 157000 of 159571
    review 158000 of 159571
    review 159000 of 159571



```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)
```


```python
train_data_features = vectorizer.fit_transform(processed_comment)
train_data_features = train_data_features.toarray()
```


```python
print(train_data_features.shape)
```

    (159571, 5000)



```python
vocab = vectorizer.get_feature_names()
```


```python
dist = np.sum(train_data_features, axis=0)
```


```python
# define model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(720, input_shape=(5000,), activation='relu'))
classifier.add(Dense(36, activation='relu'))
classifier.add(Dense(6, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

    Using TensorFlow backend.



```python
classifier.fit(train_data_features, y_train, batch_size=100, epochs=5)
```

    Epoch 1/5
    159571/159571 [==============================] - 118s 742us/step - loss: 0.0762 - acc: 0.9780
    Epoch 2/5
    159571/159571 [==============================] - 118s 740us/step - loss: 0.0489 - acc: 0.9837
    Epoch 3/5
    159571/159571 [==============================] - 118s 741us/step - loss: 0.0385 - acc: 0.9872
    Epoch 4/5
    159571/159571 [==============================] - 120s 755us/step - loss: 0.0292 - acc: 0.9907
    Epoch 5/5
    159571/159571 [==============================] - 120s 750us/step - loss: 0.0227 - acc: 0.9933





    <keras.callbacks.History at 0x7fc6344b1668>




```python
# predicting the test set result
test = pd.read_csv("test.csv")
print(test.shape)
```

    (153164, 2)



```python
num_reviews = len(test["comment_text"])
clean_test_comments = []

for i in range(0, num_reviews):
    if (i+1)%1000 == 0:
        print("comment %d of %d" %((i+1), num_reviews))
    clean_test_comments.append(comment_to_words(test["comment_text"][i]))
```

    comment 1000 of 153164
    comment 2000 of 153164
    comment 3000 of 153164
    comment 4000 of 153164
    comment 5000 of 153164
    comment 6000 of 153164
    comment 7000 of 153164
    comment 8000 of 153164
    comment 9000 of 153164
    comment 10000 of 153164
    comment 11000 of 153164
    comment 12000 of 153164
    comment 13000 of 153164
    comment 14000 of 153164
    comment 15000 of 153164
    comment 16000 of 153164
    comment 17000 of 153164
    comment 18000 of 153164
    comment 19000 of 153164
    comment 20000 of 153164
    comment 21000 of 153164
    comment 22000 of 153164
    comment 23000 of 153164
    comment 24000 of 153164
    comment 25000 of 153164
    comment 26000 of 153164
    comment 27000 of 153164
    comment 28000 of 153164
    comment 29000 of 153164
    comment 30000 of 153164
    comment 31000 of 153164
    comment 32000 of 153164
    comment 33000 of 153164
    comment 34000 of 153164
    comment 35000 of 153164
    comment 36000 of 153164
    comment 37000 of 153164
    comment 38000 of 153164
    comment 39000 of 153164
    comment 40000 of 153164
    comment 41000 of 153164
    comment 42000 of 153164
    comment 43000 of 153164
    comment 44000 of 153164
    comment 45000 of 153164
    comment 46000 of 153164
    comment 47000 of 153164
    comment 48000 of 153164
    comment 49000 of 153164
    comment 50000 of 153164
    comment 51000 of 153164
    comment 52000 of 153164
    comment 53000 of 153164
    comment 54000 of 153164
    comment 55000 of 153164
    comment 56000 of 153164
    comment 57000 of 153164
    comment 58000 of 153164
    comment 59000 of 153164
    comment 60000 of 153164
    comment 61000 of 153164
    comment 62000 of 153164
    comment 63000 of 153164
    comment 64000 of 153164
    comment 65000 of 153164
    comment 66000 of 153164
    comment 67000 of 153164
    comment 68000 of 153164
    comment 69000 of 153164
    comment 70000 of 153164
    comment 71000 of 153164
    comment 72000 of 153164
    comment 73000 of 153164
    comment 74000 of 153164
    comment 75000 of 153164
    comment 76000 of 153164
    comment 77000 of 153164
    comment 78000 of 153164
    comment 79000 of 153164
    comment 80000 of 153164
    comment 81000 of 153164
    comment 82000 of 153164
    comment 83000 of 153164
    comment 84000 of 153164
    comment 85000 of 153164
    comment 86000 of 153164
    comment 87000 of 153164
    comment 88000 of 153164
    comment 89000 of 153164
    comment 90000 of 153164
    comment 91000 of 153164
    comment 92000 of 153164
    comment 93000 of 153164
    comment 94000 of 153164
    comment 95000 of 153164
    comment 96000 of 153164
    comment 97000 of 153164
    comment 98000 of 153164
    comment 99000 of 153164
    comment 100000 of 153164
    comment 101000 of 153164
    comment 102000 of 153164
    comment 103000 of 153164
    comment 104000 of 153164
    comment 105000 of 153164
    comment 106000 of 153164
    comment 107000 of 153164
    comment 108000 of 153164
    comment 109000 of 153164
    comment 110000 of 153164
    comment 111000 of 153164
    comment 112000 of 153164
    comment 113000 of 153164
    comment 114000 of 153164
    comment 115000 of 153164
    comment 116000 of 153164
    comment 117000 of 153164
    comment 118000 of 153164
    comment 119000 of 153164
    comment 120000 of 153164
    comment 121000 of 153164
    comment 122000 of 153164
    comment 123000 of 153164
    comment 124000 of 153164
    comment 125000 of 153164
    comment 126000 of 153164
    comment 127000 of 153164
    comment 128000 of 153164
    comment 129000 of 153164
    comment 130000 of 153164
    comment 131000 of 153164
    comment 132000 of 153164
    comment 133000 of 153164
    comment 134000 of 153164
    comment 135000 of 153164
    comment 136000 of 153164
    comment 137000 of 153164
    comment 138000 of 153164
    comment 139000 of 153164
    comment 140000 of 153164
    comment 141000 of 153164
    comment 142000 of 153164
    comment 143000 of 153164
    comment 144000 of 153164
    comment 145000 of 153164
    comment 146000 of 153164
    comment 147000 of 153164
    comment 148000 of 153164
    comment 149000 of 153164
    comment 150000 of 153164
    comment 151000 of 153164
    comment 152000 of 153164
    comment 153000 of 153164



```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>comment_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00001cee341fdb12</td>
      <td>Yo bitch Ja Rule is more succesful then you'll...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000247867823ef7</td>
      <td>== From RfC == \n\n The title is fine as it is...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00013b17ad220c46</td>
      <td>" \n\n == Sources == \n\n * Zawe Ashton on Lap...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00017563c3f7919a</td>
      <td>:If you have a look back at the source, the in...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00017695ad8997eb</td>
      <td>I don't anonymously edit articles at all.</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data_features = vectorizer.transform(clean_test_comments)
test_data_features = test_data_features.toarray()
```


```python
# model prediction
result1 = classifier.predict_proba(test_data_features)
```


```python
result1[0]
```




    array([0.9999995 , 0.44507396, 0.99986875, 0.0370819 , 0.9795308 ,
           0.14327948], dtype=float32)




```python
# each category probability model prediction
output = np.column_stack([test["id"], result1])
```


```python
output[0]
```




    array(['00001cee341fdb12', 0.9999995231628418, 0.4450739622116089,
           0.9998687505722046, 0.03708190470933914, 0.9795308113098145,
           0.14327947795391083], dtype=object)




```python
output.shape
```




    (153164, 7)




```python
output = pd.DataFrame(output)
```


```python
output = output.rename(index=str, columns={0:"id", 1:"toxic", 2:"severe_toxic", 3:"obscene", 4:"threat", 5:"insult", 6:"identity_hate"})
```


```python
# Use pandas to write the comma-separated output file
output.to_csv("toxic_comment.csv", index=False, quoting=3)
```


```python
for i in range(0, num_reviews):
    if output.iloc[i]["id"] == "0114509409588767":
        print(i)
```

    647



```python
output.iloc[0]["id"]
```




    '00001cee341fdb12'

