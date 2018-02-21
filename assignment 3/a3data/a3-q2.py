from nltk.corpus import stopwords
import loader
import xml.etree.cElementTree as ET
import codecs
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import nltk
from nltk.probability import *

def remove_stopwords(s):
	for item in s.values():
		for word in item.context:
			if word in stopwords.words('english'):
				item.context.remove(word)

#def accuracy(syn, k, x, sum, count):
	#for item in syn:
			#x.append((item.key()).encode("utf-8"))
		#for key in k[entry]:
			#if key in x:
				#sum+=1
	#return float(sum)/count

def baseline(s, k):
	sum=0
	count=len(s)
	for entry in s:
		x=[]
		word=s[entry].lemma
		m=wn.synsets(word)[0].lemmas()
		for item in m:
			x.append((item.key()).encode("utf-8"))
		for key in k[entry]:
			if key in x:
				sum+=1
	return float(sum)/count

def compare_lesk(s, k):
	sum=0
	count=len(s)
	remove_stopwords(s)
	baseline_accuracy=baseline(s, k)
	for entry in s:
		x=[]
		result=(lesk(s[entry].context, s[entry].lemma)).lemmas()
		for item in result:
			x.append(item.key().encode("utf-8"))
		for key in k[entry]:
			if key in x:
				sum+=1
	lesk_accuracy = float(sum)/count
	print "Lesk accuracy: " + str(lesk_accuracy*100) +"%"+ " Baseline accuracy: " + str(baseline_accuracy*100)+"%"

def compare_modified_lesk(s, k):
	sum=0
	count=len(s)
	remove_stopwords(s)
	baseline_accuracy=baseline(s, k)
	data=[]
	for entry in s:
		x=[]
		data, results=(modified_lesk(s[entry].context, s[entry].lemma, data ))
		result=results.lemmas()
		for item in result:
			#print item.key()
			x.append(item.key().encode("utf-8"))
		for key in k[entry]:
			if key in x:
				sum+=1
	lesk_accuracy = float(sum)/count
	print "Modified Lesk accuracy: " + str(lesk_accuracy*100) +"%"+ " Baseline accuracy: " + str(baseline_accuracy*100)+"%"

def modified_lesk(context_sentence, ambiguous_word, data):
	context = set(context_sentence)
	synsets = wn.synsets(ambiguous_word)
	if not synsets:
		return None
	if not data:
		_, sense=take_max(context,synsets)
		data.append(sense)
		return data, sense
	dist=synsets
	if len(dist)==1:
		sense=synsets[0]
		data.append(sense)
		return data, sense
	fd = nltk.FreqDist(dist)
	for synset in synsets:
		x=get_occurences(synset, data)
		for element in fd:
			fd[element]=fd[element]+x
	sense=MLEProbDist(fd).max()
	data.append(sense)
	return data, sense

def take_max(context, synsets):
	return max((len(context.intersection(ss.definition().split())), ss) for ss in synsets)
def get_occurences(synset, data):
		if synset in data:
			return data.count(synset)
		else:
			return 0
			
			

data_f = 'multilingual-all-words.en.xml'
key_f = 'wordnet.en.key'
dev_instances, test_instances = loader.load_instances(data_f)
dev_key, test_key = loader.load_key(key_f)
    
# IMPORTANT: keys contain fewer entries than the instances; need to remove them
dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}
compare_modified_lesk(test_instances, test_key)
#results = lesk:29.586% baseline: 62.344% modified: 30.89%
