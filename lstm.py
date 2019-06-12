#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# load positive labels
pos = open('data/ira_10000.csv').read()
npos = 0
labels, texts = [], []
for i, line in enumerate(pos.split("\n")):
    content = line.split(',')
    if len(content) < 4:
    	continue;
    if content[4] != "English":
    	continue;
    labels.append(1)
    texts.append(content[2])
    npos += 1

# load negative labels (random tweets)
neg = open('data/tweets-2016-10000-textonly.txt').read()
nneg = 0
for i, line in enumerate(neg.split("\n")):
    labels.append(0)
    texts.append(line)
    nneg += 1

texts, labels = shuffle(texts, labels)

print('Total number of datapoints: ', len(labels))
print('Positive labels: ', npos)
print('Negative labels: ', nneg)

df = pd.DataFrame()
df['text'] = texts
df['label'] = labels

df.head()
#
#
# # In[8]:
#
#
# sns.countplot(df.label)
# plt.xlabel('Label')
# plt.title('Breakdown by label type')


# In[21]:


enc = LabelEncoder()
y = enc.fit_transform(labels)
train_x, test_x, train_y, test_y = train_test_split(df['text'], y, test_size=0.20)

train_y_array = np.empty((len(train_y), 2))
test_y_array = np.empty((len(test_y), 2))

for ii in range(len(train_y)):
    if train_y[ii] == 1:
        train_y_array[ii] = np.array([1, 0])
    else:
        train_y_array[ii] = np.array([0, 1])

for ii in range(len(test_y)):
    if test_y[ii] == 1:
        test_y_array[ii] = np.array([1, 0])
    else:
        test_y_array[ii] = np.array([0, 1])

# train_y = train_y_array
# test_y = test_y_array

# Tokenize the text. Max length 280 (may tweak)

maxlen = 280

token = Tokenizer()
token.fit_on_texts(df['text'])

sequences = token.texts_to_sequences(train_x)
padded = sequence.pad_sequences(sequences, maxlen=maxlen)


# In[25]:


def make_rnn():
    inputs = Input(name='inputs',shape=[maxlen])
    layer = Embedding(len(token.word_index)+1,50,input_length=maxlen)(inputs)
    layer = LSTM(64, dropout=0.2, return_sequences=True)(layer)
    layer = LSTM(64, dropout=0.2)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    # layer = Dense(256, name='FC2')(layer)
    # layer = Activation('relu')(layer)
    # layer = Dense(256, name='FC3')(layer)
    # layer = Activation('tanh')(layer)
    # layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = make_rnn()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

plot_model(model, to_file='model.png')


# In[26]:


history = model.fit(padded,train_y,batch_size=128,epochs=10,
          validation_split=0.2)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# In[29]:


test_sequences = token.texts_to_sequences(test_x)
test_padded = sequence.pad_sequences(test_sequences,maxlen=maxlen)
results = model.predict(test_padded)

# num_correct = 0
# total = 0

# for ii in range(len(results)):
#     result = max(results[ii])
#     pred = 0
#     if result == results[ii][0]:
#         print('troll')
#         pred = 1
#     if pred == test_y[ii]:
#         num_correct += 1
#     total += 1
#
# print('Acc: ', num_correct / total)

accuracy = model.evaluate(test_padded, test_y)
print('Test loss: ', accuracy[0])
print('Test accuracy: ', accuracy[1])


# In[ ]:





# In[ ]:




