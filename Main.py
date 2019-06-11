#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import random

from sklearn.naive_bayes import GaussianNB


# In[ ]:


f = open('data/tweets-2016-1000-textonly.txt', 'r')
lines = f.readlines()

tweets = []
labels = []

len_train = 1000


# In[ ]:


with open('data/IRAhandle_tweets_1.csv', newline='') as csvfile:
    categories = csvfile.readline().split(",")
    tweetreader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in tweetreader:
        tweet = dict(zip(categories, row))
        if tweet['language'] == 'English':
            tweets.append(tweet['content'])
            labels.append(1)
            counter += 1
        if counter > len_train:
            break
csvfile.close()

# In[ ]:

for line in lines:
    # for line in lines:
    #     tweet = json.loads(line)
    #     if 'user' in tweet.keys():
    #         if tweet["user"]["lang"] == "en":
    #             tweets.append(tweet['text'])
    #             labels.append(0)
    tweets.append(line)
    labels.append(0)

f.close()
            
tweets_to_labels = dict(zip(tweets, labels))
random.shuffle(tweets)

actual = []

for tweet in tweets:
    actual.append(tweets_to_labels[tweet])

# In[ ]:


vectorizer = CountVectorizer(binary=True, lowercase=True)
total = vectorizer.fit_transform(np.array(tweets))
train = total[:len_train]
test = total

model = GaussianNB()
model.fit(train.toarray(), actual[:len_train])
predicted_troll = model.predict(test.toarray())


# In[ ]:


correct = 0
true_positive = 0
total_positive = 0
true_negative = 0
total_negative = 0

for ii in range(len(predicted_troll)):
    if actual[ii]:
        total_positive += 1
    else:
        total_negative += 1
    if predicted_troll[ii] == actual[ii]:
        correct += 1
        if actual[ii]:
            true_positive += 1
        else:
            true_negative += 1
        
# accuracy
print('Total accuracy:')
print(correct / len(predicted_troll))
print('Positive label accuracy:')
print(true_positive / total_positive)
print('Negative label accuracy:')
print(true_negative / total_negative)


# In[ ]:





# In[ ]:




