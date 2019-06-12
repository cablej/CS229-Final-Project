# Outline from https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text as text2
from sklearn import decomposition, ensemble

import pandas, numpy as np, textblob, string
from sklearn.utils import shuffle
import operator as op

import warnings

if __name__ == "__main__":
	nb = False
	lr = False
	svm_ = True
	rf = True
else:	
	nb = False
	lr = False
	svm_ = False
	rf = False

warnings.simplefilter(action='ignore', category=FutureWarning)

def train_model(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False, should_do_common=False):
	# fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)
	
	train_preds = classifier.predict(feature_vector_train)
	test_preds = classifier.predict(feature_vector_test)

	# stop_words = text2.ENGLISH_STOP_WORDS.union(["http", "https", "amp", "amb"])
	
	if is_neural_net:
		test_preds = test_preds.argmax(axis=-1)

	if should_do_common:
	    feature_names = count_vect.get_feature_names()
	    diff = classifier.feature_log_prob_[1,:] - np.max(classifier.feature_log_prob_[0:])

	    name_diff = {}
	    for i in range(len(feature_names)):
	       name_diff[feature_names[i]] = diff[i]

	       names_diff_sorted = sorted(name_diff.items(), key = op.itemgetter(1), reverse = True)
	    c = 0
	    i = 0
	    while c < 50:
	       if names_diff_sorted[i][0] in stop_words or len(names_diff_sorted[i][0]) <= 2:
	       	 i += 1
	       	 continue
	       print(names_diff_sorted[i])
	       c += 1
	       i += 1
	
	train_acc = metrics.accuracy_score(train_preds, train_y)
	test_acc = metrics.accuracy_score(test_preds, test_y)
	cm = metrics.confusion_matrix(test_y, test_preds)
	print('Train Accuracy: ', train_acc)
	print('Test Accuracy: ', test_acc)
	print('Confusion matrix: ', cm)
	return (test_acc, cm)

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

train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size=0.8)

print('Size of training set: ', len(train_x))
print('Size of Test set:', len(test_x))

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

binary_count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=True)
binary_count_vect.fit(trainDF['text'])

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

binary_xtrain_count = binary_count_vect.transform(train_x)
binary_xtest_count = binary_count_vect.transform(test_x)

xtrain_count =  count_vect.transform(train_x)
xtest_count =  count_vect.transform(test_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x) 

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
	accuracy = train_model(naive_bayes.MultinomialNB(), binary_xtrain_count, train_y, binary_xtest_count)

	# Naive Bayes on Count Vectors
	print("NB, Count Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count)

	# Naive Bayes on Word Level TF IDF Vectors
	print("NB, WordLevel TF-IDF: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)

	# Naive Bayes on Ngram Level TF IDF Vectors
	print("NB, N-Gram Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)

	# Naive Bayes on Character Level TF IDF Vectors
	print("NB, CharLevel Vectors: ")
	accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)

if lr:

	# LR on Count Vectors
	print("LR, Binary Count Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), binary_xtrain_count, train_y, binary_xtest_count)

	# Linear Classifier on Count Vectors
	print("LR, Count Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count)

	# Linear Classifier on Word Level TF IDF Vectors
	print("LR, WordLevel TF-IDF: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf)

	# Linear Classifier on Ngram Level TF IDF Vectors
	print("LR, N-Gram Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)

	# Linear Classifier on Character Level TF IDF Vectors
	print("LR, CharLevel Vectors: ")
	accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)

if svm_:

	def svm_tune(x, y):
		Cs = [0.001, 0.01, 0.1, 1, 10]
		gammas = [0.001, 0.01, 0.1, 1]
		grid = {'C': Cs, 'gamma': gammas}
		search = GridSearchCV(svm.SVC(kernel='rbf'), grid)
		search.fit(x, y)
		search.best_params_
		return search.best_params_

	# print(svm_tune(xtrain_count, train_y))
	# print(svm_tune(xtrain_tfidf, train_y))
	# print(svm_tune(xtrain_tfidf_ngram, train_y))

	# SVM on Bin Count Vectors
	print("SVM, Binary Count Vectors: ")
	accuracy = train_model(svm.SVC(kernel='rbf', C=10, gamma=0.1), binary_xtrain_count, train_y, binary_xtest_count)

	# SVM on Count Vectors
	print("SVM, Count Vectors: ")
	accuracy = train_model(svm.SVC(kernel='rbf', C=10, gamma=0.1), xtrain_count, train_y, xtest_count)

	# SVM on Word Level TF IDF Vectors
	print("SVM, WordLevel TF-IDF: ")
	accuracy = train_model(svm.SVC(kernel='rbf', C=10, gamma=0.1), xtrain_tfidf, train_y, xtest_tfidf)

	# SVM on Ngram Level TF IDF Vectors
	print("SVM, N-Gram Vectors: ")
	accuracy = train_model(svm.SVC(kernel='rbf', C=10, gamma=0.1), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)

	# SVM on Character Level TF IDF Vectors
	print("SVM, CharLevel Vectors: ")
	accuracy = train_model(svm.SVC(kernel='rbf', C=10, gamma=0.1), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)

if rf:

	# SVM on Bin Count Vectors
	print("RF, Binary Count Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), binary_xtrain_count, train_y, binary_xtest_count)

	# RF on Count Vectors
	print("RF, Count Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xtest_count)

	# RF on Word Level TF IDF Vectors
	print("RF, WordLevel TF-IDF: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf)

	# RF on Ngram Level TF IDF Vectors
	print("RF, N-Gram Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)

	# RF on Character Level TF IDF Vectors
	print("RF, CharLevel Vectors: ")
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
