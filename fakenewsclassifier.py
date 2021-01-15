# -*- coding: utf-8 -*-
"""FakeNewsClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q7Gj0coYB-qTGLErZj8BetTZUw1l7ht0
"""

from google.colab import drive
drive.mount("/content/gdrive")

import pandas as pd
df=pd.read_csv('/content/gdrive/My Drive/Datasets/train.csv')
df_test=pd.read_csv('/content/gdrive/My Drive/Datasets/test.csv')

print(df.head())
df_test.fillna("missing",inplace=True)

# drop nan values
df=df.dropna()

# get the independent features
X=df.drop('label',axis=1)

# get the dependent feature
y=df['label']

print(X.shape)
print("\n",y.shape)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
# %tensorflow_version 2.x
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

# vocabulary size
voc_size=5000

messages=X.copy()
messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

## data preprocessing 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
  review=re.sub('[^a-zA-z]',' ',messages['title'][i])
  review=review.lower()  # to make each word lower
  review=review.split()  # list of words 

  review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
  review=' '.join(review)   # combining the words
  corpus.append(review)     # appending in the corpus

# data preprocessing for test dataset
corpus_test=[]
test_features=df_test.copy()
for i in range(0,len(test_features)):
  review=re.sub('[^a-zA-z]',' ',test_features['title'][i])
  review=review.lower()  # to make each word lower
  review=review.split()  # list of words 

  review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
  review=' '.join(review)        # combining the words
  corpus_test.append(review)     # appending in the corpus_test

onhot_repr=[one_hot(words,voc_size) for words in corpus]

# to make each sentence of fixed length we use pad sequence
sent_length=20
embedded_docs=pad_sequences(onhot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

# for test 
onhot_repr_test=[one_hot(words,voc_size) for words in corpus_test]

# to make each sentence of fixed length we use pad sequence
embedded_docs_test=pad_sequences(onhot_repr_test,padding='pre',maxlen=sent_length)
print(embedded_docs_test)

# creating model
from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

import numpy as np 
X_final=np.array(embedded_docs)
y_final=np.array(y)

X_final_test=np.array(embedded_docs_test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=42)

# model training 
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# performance metrix and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=model.predict_classes(X_test)
y_pred_test=model.predict_classes(X_final_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

val=[]
for i in y_pred_test:
    val.append(i[0])

IDtest = df_test["id"]
submissions=pd.DataFrame({"id": IDtest,
                         "Label": val})
submissions.to_csv("submission.csv", index=False, header=True)

! cat submission.csv