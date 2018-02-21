from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import sys

wordnet_lemmatizer = WordNetLemmatizer()

#this simply loads the cluster which is to be summarized
def load_files(cl):
	title=list(cl)
	doc1=open(replace_star(title, 1), "r")
	doc2=open(replace_star(title, 2), "r")
	doc3=open(replace_star(title, 3), "r")
	t_1 = doc1.read()
	t_2 = doc2.read()
	t_3 = doc3.read()
	return [t_1, t_2, t_3], title[10]

#this assists in opening the correct files
def replace_star(l,n):
	l[12]=str(n)
	output="".join(l)
	return output

#this performs general preprocessing
def preprocess(sents):
	cleaner_sentences=''.join(i for i in sents if ord(i)<128)
	data=sent_tokenize(cleaner_sentences)
	clean_data=[]
	for sentence in data:
		clean_data.append((sentence, clean(sentence)))
	return clean_data

#this method 'cleans' a sentence by getting rid of stopwords, lemmatizing each word and putting everything in lower case
def clean(s):
	output=[]
	tokens=word_tokenize(s)
	regex = re.compile('[^a-zA-Z]')
	for word in tokens:
		#remove stopwords and grammar
		w=regex.sub('', word)
		if w.lower() not in stopwords.words('english'):
			output.append(wordnet_lemmatizer.lemmatize(w.lower()))
		else:
			continue
	return output

#this gets the probabilities for each word appearing in a cluster
def get_probs(sents, counts):
	for t in sents:
		sentence=t[1]
		for word in sentence:
			if word in counts:
				counts[word]+=1
			else:
				counts[word]=1
	return normalized(counts)

#this simply normalizes counts to form a probability distribution
def normalized(d):
	l=len(d)
	for entry in d:
		d[entry]=float(d[entry])/l
	return d

#this calculates the probabilities of each sentence for the SumBasic methods
def sentence_probabilities(sents, counts):
	output={}
	n=0
	for sent in sents:
		sentence=sent[1]
		total=0
		for word in sentence:
			total+=counts[word]
		output[sent[0]]=total/len(sentence)
		n+=1
	return output

#this outputs the sorted list of sentences for the SumBasic methods
def sorted_sentences(files):
	c=[]
	for file in files:
		c.extend(preprocess(file))
	probs=get_probs(c, {})
	sent_probs=sentence_probabilities(c , probs)
	sorted_sents= sorted(sent_probs.iteritems(), key=lambda (k,v): (v,k))
	return c, sorted_sents, probs

#overall method that performs the summarization
def sumbasic():
	version=sys.argv[1]
	files, cluster_number=load_files(sys.argv[2])
	to_write=open((sys.argv[1]+"-"+ str(cluster_number) + ".txt"), 'w')
	if version=="leading":
		leading(files, cluster_number, to_write)
	else:
		sentence_t, sentences, probabilities=sorted_sentences(files)
		word_count=0
		while (word_count<100):
			temp=sentences[-1][0]
			s=temp.split(' ')
			if (word_count+len(s)<100):
				to_write.write(temp + ' ')
				sentences.pop()
				word_count+=len(s)
				if version=="orig":
					sentences, probabilities=update(sentence_t, temp, probabilities)
			else:
				break

#update method for the original SumBasic method
def update(sentence_tuples, target, probs):
	sent=[]
	for entry in sentence_tuples:
		if entry[0]==target:
			sent.extend(entry[1])
			for word in entry[1]:
				value=probs[word]
				probs[word]=value*value
	sent_probs=sentence_probabilities(sentence_tuples, probs)
	sorted_sents= sorted(sent_probs.iteritems(), key=lambda (k,v): (v,k))
	return sorted_sents, probs
		
#leading method
def leading(files, cluster_number, to_write):
	leading_file=files[0]
	clean_file=''.join(i for i in leading_file if ord(i)<128)
	word_count=0
	to_print=""
	sents=sent_tokenize(clean_file)
	for sent in sents:
		s=sent.split(' ')
		if (word_count+len(s)<100):
			to_write.write(sent + ' ')
			word_count+=len(s)
		else:
			break

sumbasic()
