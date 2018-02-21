#vectorize all the data sets
#train the classifier on train
#experiment with parameters on dev to find the best model
#test on test
#compare to a random baseline
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn import datasets
from pandas import DataFrame
import numpy
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import itertools
from sklearn.linear_model import LogisticRegression
import os
from sklearn.svm import SVC
from numpy import random


POS='pos'
NEG='neg'


def build_data_frame(path, classification):
	rows=[]
	f = open(path, 'r')
	lines=[];
	tokens = sent_tokenize(repr(f.read()))
	for token in tokens:
		lines.append(token)
	f.close()
	for line in lines:
		rows.append({'text': line, 'class': classification})
	data_frame=DataFrame(rows)
	return data_frame

data = DataFrame({'text': [], 'class': []})

data=data.append(build_data_frame('rt-polarity.pos', POS))
data=data.append(build_data_frame('rt-polarity.neg', NEG))

RANGE=[(1,1), (1,2)]
STOP=[None, 'english']
CLF=[MultinomialNB(), LogisticRegression(), SVC(kernel='linear')]
MAX_DF=[.9]
MIN_DF=[.1]

params=[RANGE, STOP, CLF, MAX_DF, MIN_DF]
param_combos=list(itertools.product(*params))

results=open('results.txt', 'w')
highest_results=open('winners.txt', 'w')
best_log_reg=[0, None, 1, 1]
best_mnb=[0, None, 1, 1]
best_svc=[0, None, 1, 1]

def train_and_test(r, stop_word, clf, max_df, min_df):
	pipeline = Pipeline([
    	('count_vectorizer',   CountVectorizer(ngram_range=r, stop_words=stop_word, max_df=max_df, min_df=min_df)),
    	('classifier',         clf)
	])

	k_fold = KFold(n=len(data), n_folds=6)
	scores = []
	random_scores = []
	confusion = numpy.array([[0, 0], [0, 0]])
	for train_indices, test_indices in k_fold:
		train_text = data.iloc[train_indices]['text'].values
		train_y = data.iloc[train_indices]['class'].values.astype(str)
		test_text = data.iloc[test_indices]['text'].values
		test_y = data.iloc[test_indices]['class'].values.astype(str)
		pipeline.fit(train_text, train_y)
		random_predictions=[];
		for text in test_text:
			choice=random.choice(['pos', 'neg'])
			random_predictions.append(choice)
		predictions = pipeline.predict(test_text)
		random_score=accuracy_score(test_y, random_predictions)
		confusion += confusion_matrix(test_y, predictions)
		score = accuracy_score(test_y, predictions)
		scores.append(score)
		random_scores.append(random_score)
	if(sum(scores)/len(scores)>best_svc[0] and clf==SVC(kernel='linear')):
		best_svc[0]=sum(scores)/len(scores)
		best_svc[1]=r
		best_svc[2]=stop_word
		best_svc[3]= max_df
		best_svc[4]= min_df 
	if(sum(scores)/len(scores)>best_log_reg[0] and clf==LogisticRegression()):
		best_log_reg[0]=sum(scores)/len(scores)
		best_log_reg[1]=r
		best_log_reg[2]=stop_word
		best_log_reg[3]= max_df
		best_log_reg[4]= min_df 
	if(sum(scores)/len(scores)>best_mnb[0] and clf==MultinomialNB()):
		best_mnb[0]=sum(scores)/len(scores)
		best_svc[1]=r
		best_svc[2]=stop_word
		best_svc[3]= max_df
		best_svc[4]= min_df 
	results.write('Parameters: ' + str(r) + '\n' + str(stop_word) + '\n' + str(clf) + '\n' + str(max_df)  + '\n' + str(min_df) + '\n')
	results.write('Score:' + str(sum(scores)/len(scores)) + '\n')
	results.write('Confusion matrix:' + str(confusion))
	results.write('Random baseline:' + str(sum(random_scores)/len(random_scores)))
	results.write('\n\n\n')
	highest_results.write('SVC: ' + str(best_svc) + '\n' + 'Log Reg: ' + str(best_log_reg) + '\n' + 'MNB: ' + str(best_mnb))

for parameter_set in param_combos:
	print 'running...'
	train_and_test(parameter_set[0], parameter_set[1], parameter_set[2], parameter_set[3], parameter_set[4])
	
print 'Complete!'
