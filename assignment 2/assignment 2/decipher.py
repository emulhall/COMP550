import nltk
from nltk.tag import hmm
import sys
from nltk.tokenize import sent_tokenize
from nltk.probability import LaplaceProbDist
from nltk.probability import FreqDist
import string
import numpy
from nltk.probability import (FreqDist, ConditionalFreqDist, ConditionalProbDist, DictionaryProbDist, MLEProbDist)

def loadFiles(cipher):
	train_cipher = open("a2data/"+cipher+"/train_cipher.txt", "r")
	train_plain = open("a2data/"+cipher+"/train_plain.txt", "r")
	test_cipher = open("a2data/"+cipher+"/test_cipher.txt", "r")
	test_plain = open("a2data/"+cipher+"/test_plain.txt", "r")
	return {"train_data": train_cipher, "train_txt": train_plain, "test_data": test_cipher, "test_txt": test_plain}


def label(train, plain):
	train_tokens=train.split("\n")
	plain_tokens=plain.split("\n")
	train_data=[]
	plain_data=[]
	for sent in train_tokens:
		temp=list(sent)
		train_data.append(temp)
	for sent in plain_tokens:
		temp=list(sent)
		plain_data.append(temp)
	zipped = zip(train_data, plain_data)
	final_data=[]
	for element in zipped:
		final_data.append(zip(element[0], element[1]))
	return final_data

def testPrep(cipher):
	tokens=cipher.split("\n")
	data=[]
	for sent in tokens:
		temp=list(sent)
		data.append(temp)
	return data

def accuracy(test_output, actual):
	i=0
	correct=0
	decipher=""
	while i<len(test_output) and i<len(actual):
		decipher+=test_output[i][1]
		if test_output[i][1]==actual[i]:
			correct+=1
			i=i+1
		else:
			i=i+1
	print decipher
	return (float(correct)/float(len(actual)))
		
def test():
	files=loadFiles(sys.argv[1])
	labeled_data=label(files["train_data"].read(), files["train_txt"].read())
	trainer = hmm.HiddenMarkovModelTrainer()
	if len(sys.argv) > 2:
		if sys.argv[2]=="-laplace":
			tagger = trainer.train_supervised(labeled_data, LaplaceProbDist)
	else:
		tagger=trainer.train_supervised(labeled_data)
	test_data=testPrep(files["test_data"].read())
	comparison=testPrep(files["test_txt"].read())
	results=0
	for element in test_data:
		output=tagger.tag(element)
		results+=accuracy(output, comparison[0])
		comparison.pop(0)
	return results/len(test_data)
	
def transitionProb(sentences):
	transitions = ConditionalFreqDist()
	for sent in sentences:
		lasts=None
		for token in sent:
			if lasts is None:
				pass
			else:
				transitions[lasts][token] += 1
			lasts = token
	return transitions

def preprocess():
	pos = open("rt-polaritydata/rt-polarity.pos", "r")
	neg = open("rt-polaritydata/rt-polarity.neg", "r")
	pos_tokens=testPrep(pos.read())
	neg_tokens=testPrep(neg.read())
	total_tokens=numpy.concatenate([pos_tokens, neg_tokens])
	processed=passage_processing(total_tokens)
	return transitionProb(processed)

def passage_processing(tokens):
	symbols=list(string.ascii_lowercase)
	symbols.append(' ')
	symbols.append(',')
	symbols.append('.')
	result=[]
	for sent in tokens:
		add=[]
		for letter in sent:
			if letter in symbols:
				add.append(letter.lower())
			else:
				pass
		result.append(add[0:-1])
	return result

def train_transitions(labelled_sequences, additional_transitions, estimator=None):
    # default to the MLE estimate
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)

    # count occurrences of starting states, transitions out of each state
    # and output symbols observed in each state
    known_symbols = []
    known_states = []

    starting = FreqDist()
    transitions = ConditionalFreqDist()
    outputs = ConditionalFreqDist()
    for sequence in labelled_sequences:
        lasts = None
        for token in sequence:
            state = token[0]
            symbol = token[1]
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            outputs[state][symbol] += 1
            lasts = state

            # update the state and symbol lists
            if state not in known_states:
                known_states.append(state)

            if symbol not in known_symbols:
                known_symbols.append(symbol)

    # create probability distributions (with smoothing)
    N = len(known_states)
    pi = estimator(starting, N)
    A = ConditionalProbDist(ConditionalFreqDist.__add__(transitions, additional_transitions) , estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(known_symbols))
    return hmm.HiddenMarkovModelTagger(known_states, known_symbols, A, B, pi)

def test_with_transitions(dist):
	files=loadFiles(sys.argv[1])
	labeled_data=label(files["train_data"].read(), files["train_txt"].read())
	if dist==1:
		trainer = train_transitions(labeled_data, preprocess(), LaplaceProbDist)
	else:
		trainer = train_transitions(labeled_data, preprocess())
	test_data=testPrep(files["test_data"].read())
	comparison=testPrep(files["test_txt"].read())
	results=0
	for element in test_data:
		output=trainer.tag(element)
		results+=accuracy(output, comparison[0])
		comparison.pop(0)
	return results/len(test_data)

if len(sys.argv)>2:
	if len(sys.argv)==3:
		if sys.argv[2]=="-lm":
			print test_with_transitions(0)
		else:
			print test()
	elif len(sys.argv)>3:
		if sys.argv[3]=="-lm" or sys.argv[2]=="-lm":
			print test_with_transitions(1)
		else:
			print test()
	else:
		print test()
else:
	print test()
