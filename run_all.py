# Most of this is from https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text as text2
from sklearn import decomposition, ensemble

import pandas, numpy as np, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.utils import shuffle
import operator as op

import warnings

if __name__ == "__main__":
	nb = True
	lr = True
	svm_ = True
	rf = True
	shallow_nn = False
	deep_nn = False
else:	
	nb = False
	lr = False
	svm_ = False
	rf = False
	shallow_nn = False
	deep_nn = False

warnings.simplefilter(action='ignore', category=FutureWarning)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False, should_do_common=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    train_preds = classifier.predict(feature_vector_train)
    valid_preds = classifier.predict(feature_vector_valid)

    stop_words = text2.ENGLISH_STOP_WORDS.union(["http", "https", "amp", "amb"])
    
    if is_neural_net:
        valid_preds = valid_preds.argmax(axis=-1)

    # if should_do_common:
	   #  feature_names = count_vect.get_feature_names()
	   #  diff = classifier.feature_log_prob_[1,:] - np.max(classifier.feature_log_prob_[0:])

	   #  name_diff = {}
	   #  for i in range(len(feature_names)):
	   #     name_diff[feature_names[i]] = diff[i]

	   #     names_diff_sorted = sorted(name_diff.items(), key = op.itemgetter(1), reverse = True)
	   #  c = 0
	   #  i = 0
	   #  while c < 50:
	   #     if names_diff_sorted[i][0] in stop_words or len(names_diff_sorted[i][0]) <= 2:
	   #     	 i += 1
	   #     	 continue
	   #     print(names_diff_sorted[i])
	   #     c += 1
	   #     i += 1
    
    train_acc = metrics.accuracy_score(train_preds, train_y)
    valid_acc = metrics.accuracy_score(valid_preds, valid_y)
    cm = metrics.confusion_matrix(valid_y, valid_preds)
    print('Train Accuracy: ', train_acc)
    print('Validation Accuracy: ', valid_acc)
    # print('Confusion matrix: ', cm)
    return (valid_acc, cm)

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

trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

print('Size of training set: ', len(train_x))
print('Size of validation set:', len(valid_x))

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

binary_count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=True)
binary_count_vect.fit(trainDF['text'])

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

binary_xtrain_count = binary_count_vect.transform(train_x)
binary_xvalid_count = binary_count_vect.transform(valid_x)

xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

if deep_nn:
	# load the pre-trained word-embedding vectors 
	embeddings_index = {}
	for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
	    values = line.split()
	    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

	# create a tokenizer 
	token = text.Tokenizer(num_words=5000)
	token.fit_on_texts(trainDF['text'])
	word_index = token.word_index

	# convert text to sequence of tokens and pad them to ensure equal length vectors 
	train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
	valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

	# create token-embedding mapping
	embedding_matrix = np.zeros((len(word_index) + 1, 300))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector

# API function for server to predict using nb (using count vectors)
nb_classifier = naive_bayes.MultinomialNB()
nb_classifier.fit(xtrain_count, train_y)
def nb_predict(tweet):
	sample = count_vect.transform([tweet])
	return (nb_classifier.predict(sample), nb_classifier.predict_proba(sample))

# API function for server to preidct using lr (using charlevel TF-IDF vecotrs)
lr_classifier = linear_model.LogisticRegression()

lr_classifier.fit(xtrain_tfidf_ngram_chars, train_y)
def lr_predict(tweet):
	sample = tfidf_vect_ngram_chars.transform([tweet])
	return (lr_classifier.predict(sample), lr_classifier.predict_proba(sample))

if nb:


	# Naive Bayes on Count Vectors
	print("NB, Binary Count Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), binary_xtrain_count, train_y, binary_xvalid_count)

	# Naive Bayes on Count Vectors
	print("NB, Count Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)

	# Naive Bayes on Word Level TF IDF Vectors
	print("NB, WordLevel TF-IDF: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

	# Naive Bayes on Ngram Level TF IDF Vectors
	print("NB, N-Gram Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

	# Naive Bayes on Character Level TF IDF Vectors
	print("NB, CharLevel Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

if lr:
	# Linear Classifier on Count Vectors
	print("LR, Count Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)

	# Linear Classifier on Word Level TF IDF Vectors
	print("LR, WordLevel TF-IDF: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)

	# Linear Classifier on Ngram Level TF IDF Vectors
	print("LR, N-Gram Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

	# Linear Classifier on Character Level TF IDF Vectors
	print("LR, CharLevel Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

# if svm_:


	# # SVM on Count Vectors
	# print("SVM, Count Vectors: ")
	# accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)

	# # SVM on Word Level TF IDF Vectors
	# print("SVM, WordLevel TF-IDF: ")
	# accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)

	# # SVM on Ngram Level TF IDF Vectors
	# print("SVM, N-Gram Vectors: ")
	# accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

	# # SVM on Character Level TF IDF Vectors
	# print("SVM, CharLevel Vectors: ")
	# accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

if rf:
	# RF on Count Vectors
	print("SVM, Count Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)

	# RF on Word Level TF IDF Vectors
	print("SVM, WordLevel TF-IDF: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)

	# RF on Ngram Level TF IDF Vectors
	print("SVM, N-Gram Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

	# RF on Character Level TF IDF Vectors
	print("SVM, CharLevel Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

# if shallow_nn:
# 	def create_model_architecture(input_size):
# 	    # create input layer 
# 	    input_layer = layers.Input((input_size, ), sparse=True)
	    
# 	    # create hidden layer
# 	    hidden_layer = layers.Dense(400, activation="relu")(input_layer)
	    
# 	    # create output layer
# 	    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

# 	    classifier = models.Model(inputs = input_layer, outputs = output_layer)
# 	    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
# 	    return classifier 

# 	classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
# 	print("NN, Ngram Level TF IDF Vectors")
# 	accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)

# if deep_nn:

# 	def create_cnn():
# 	    # Add an Input Layer
# 	    input_layer = layers.Input((70, ))

# 	    # Add the word embedding Layer
# 	    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
# 	    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

# 	    # Add the convolutional Layer
# 	    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

# 	    # Add the pooling Layer
# 	    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

# 	    # Add the output Layers
# 	    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
# 	    output_layer1 = layers.Dropout(0.25)(output_layer1)
# 	    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

# 	    # Compile the model
# 	    model = models.Model(inputs=input_layer, outputs=output_layer2)
# 	    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
	    
# 	    return model

# 	classifier = create_cnn()
# 	print("CNN, Word Embeddings")
# 	accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
