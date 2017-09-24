import random
import math
import re
import matplotlib.pyplot as plt
import numpy as np

def tokenise(sentence):
	return sentence.split(' ')

def tokenise_corpus(corpus):
    return [tokenise(sentence) for sentence in corpus]

def token_counts(corpus):
    tokens = {}
    for sentence in corpus:
        for word in sentence:
            if word not in tokens:
                tokens[word] = 0
            tokens[word] += 1
    return tokens

def get_bigrams(corpus):
    bigrams = {}
    for sentence in corpus:
        for index, word in enumerate(sentence):
            if index > 0:
                pair  = (sentence[index - 1], word)
                if pair not in bigrams:
                    bigrams[pair] =0
                bigrams[pair] += 1
    return bigrams

def get_trigrams(corpus):
	trigrams = {}
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in trigrams:
					trigrams[triplet] = 0
				trigrams[triplet] += 1
	return trigrams

rawcorpus = [x.rstrip('\n') for x in open('./corpora/movies.txt') ]
corpus = tokenise_corpus(rawcorpus)
tokens = token_counts(corpus)

						##Unigram model:

def unigramprob(uni_prob,corpus):
	for sentence in corpus:
		for word in sentence:
			if word not in uni_prob:
				uni_prob[word] = 0
			uni_prob[word] = tokens[word]/(len(corpus) * 1.0)
	return uni_prob

uni_prob = {}
uni_prob = unigramprob(uni_prob,corpus)
#print uni_prob

						##Bigram model:

def bigramprob(bi_prob,bigrams,corpus):
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 0:
				pair = (sentence[index-1],word)
				if pair not in bi_prob:
					bi_prob[pair] = 0
				bi_prob[pair] = bigrams[pair]/(tokens[sentence[index-1]] * 1.0)
			#print tokens[sentence[index-1]]
	return bi_prob

bi_prob = {}
bigrams = get_bigrams(corpus)
bi_prob = bigramprob(bi_prob,bigrams,corpus)
#print bi_prob

						##Trigram model:

def trigramprob(tri_prob,trigrams,corpus):
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in tri_prob:
					tri_prob[triplet] = 0
				pair = (sentence[index-2], sentence[index-1])
				tri_prob[triplet] = trigrams[triplet]/(bigrams[pair]*1.0)
	return tri_prob

tri_prob = {}
trigrams = get_trigrams(corpus)
tri_prob = trigramprob(tri_prob,trigrams,corpus)
#print tri_prob

						##Plotting zipf and log-log curve:

zipf = 'zipf'
log = 'log'
def plotcurves(sorted_ngrams,flag,graph):
	sortedoccur = []
	sortedoccurlog = []
	for key, item in sorted_ngrams:
		sortedoccur.append(item)
		if(item <> 0):
			sortedoccurlog.append(math.log(item))
		else:
			sortedoccurlog.append(math.log(1))
		#Zipf curve:
	if graph is zipf:
		uni_x = sortedoccur
		plt.plot(uni_x)
		plt.ylabel('Occurence of tokens')
		plt.xlabel('Words sorted according to frequency')
		if flag == 0 :
			plt.show()
		#Log-log curve:
	if graph is log:	
		uni_x = sortedoccurlog
		plt.plot(uni_x)
		plt.ylabel('Log of occurence of tokens')
		plt.xlabel('Words sorted according to frequency')
		if flag == 0 :	
			plt.show()

		#For unigrams:
sorted_tokens = sorted(tokens.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_tokens,0,zipf)
#plotcurves(sorted_tokens,0,log)

		#For bigrams:
sorted_bigrams = sorted(bigrams.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_bigrams,0,zipf)
#plotcurves(sorted_bigrams,0,log)

		#For trigrams:
sorted_trigrams = sorted(trigrams.items(),key = lambda x: x[1], reverse=True)
#plotcurves(sorted_trigrams,0,zipf)
#plotcurves(sorted_trigrams,0,log)

						##Laplace Smoothing:
		#For bigrams:

def bigramprobls(bi_prob_ls,bigrams,tokens,corpus):
	V_bi = len(tokens.items())
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 0:
				pair = (sentence[index-1],word)
				if pair not in bi_prob_ls:
					bi_prob_ls[pair] = 0
				bi_prob_ls[pair] = (bigrams[pair]+1)/(float)(tokens[sentence[index-1]] + V_bi)
	return bi_prob_ls

bi_prob_ls = {}
bi_prob_ls = bigramprobls(bi_prob_ls,bigrams,tokens,corpus)
#print bi_prob_ls

		#For trigrams:

def trigramprobls(tri_prob_ls,trigrams,bigrams,corpus):
	V_tri = len(bigrams.items())
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in tri_prob_ls:
					tri_prob_ls[triplet] = 0
				pair = (sentence[index-2], sentence[index-1])
				tri_prob_ls[triplet] = (trigrams[triplet]+1)/(float)(bigrams[pair] + V_tri)
	return tri_prob_ls

tri_prob_ls = {}
tri_prob_ls = trigramprobls(tri_prob_ls,trigrams,bigrams,corpus)
#print tri_prob_ls

						##Plotting Laplace smoothed curves:
		#For bigrams:

def getbigrams_ls(bigrams_ls,bigrams,bi_prob_ls,corpus):
	for sentence in corpus:
			for index, word in enumerate(sentence):
				if index > 0:
					pair = (sentence[index-1],word)
					if pair not in bigrams_ls:
						bigrams_ls[pair] = 0
					bigrams_ls[pair] = bi_prob_ls[pair] * tokens[sentence[index-1]]
	return bigrams_ls

bigrams_ls = {}
bigrams_ls = getbigrams_ls(bigrams_ls,bigrams,bi_prob_ls,corpus)
#print bigrams_ls

		#Plotting:
sorted_bigrams_ls = sorted(bigrams_ls.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_bigrams_ls,0,zipf)
#plotcurves(sorted_bigrams_ls,0,log)

		#For trigrams:

def gettriigrams_ls(trigrams_ls,trigrams,tri_prob_ls,bigrams,corpus):
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in trigrams_ls:
					trigrams_ls[triplet] = 0
				pair = (sentence[index-2], sentence[index-1])
				trigrams_ls[triplet] = tri_prob_ls[triplet] * bigrams[pair]
	return trigrams_ls

trigrams_ls = {}
trigrams_ls = gettriigrams_ls(trigrams_ls,trigrams,tri_prob_ls,bigrams,corpus)
#print trigrams_ls

		#Plotting:
sorted_trigrams_ls = sorted(trigrams_ls.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_trigrams_ls,0,zipf)
#plotcurves(sorted_trigrams_ls,0,log)

						##Witten-Bell backoff:
		#For unigrams:

def unigramprob_wb(uni_prob_wb,unigramprob):
	uni_prob_wb = uni_prob
	return uni_prob_wb

uni_prob_wb = {}
uni_prob_wb = unigramprob_wb(uni_prob_wb,unigramprob)

		#For bigrams:

def bigramprob_wb(bigrams,bi_prob,bi_prob_wb,uni_prob_wb):
	count_word_bi = {}
	for pair in bigrams:
		(temp1,temp2) = pair
		if temp2 not in count_word_bi:
			count_word_bi[temp2] = 0
		count_word_bi[temp2] += 1
		summ = 0
	for key in count_word_bi:
		summ += count_word_bi[key]
	lamda_bi = 1 - len(count_word_bi)/(float)((len(count_word_bi) + summ))
	for pair in bigrams:
		if pair not in bi_prob_wb:
			bi_prob_wb[pair] = 0
		(temp1,temp2) = pair
		bi_prob_wb[pair] = lamda_bi * bi_prob[pair] + (1-lamda_bi)*(uni_prob_wb[temp2])
	return bi_prob_wb

bi_prob_wb = {}
bi_prob_wb = bigramprob_wb(bigrams,bi_prob,bi_prob_wb,uni_prob_wb)
#print bi_prob_wb

		#For trigrams:

def trigramprob_wb(trigrams,tri_prob,tri_prob_wb,bi_prob_wb):
	count_word_tri = {}
	for triplet in trigrams:
		(temp1,temp2,temp3) = triplet
		if temp3 not in count_word_tri:
			count_word_tri[temp3] = 0
		count_word_tri[temp3] += 1
		summ = 0
		for key in count_word_tri:
			summ += count_word_tri[key]
	lamda_tri = 1 - len(count_word_tri)/(float)((len(count_word_tri)+summ))
	for triplet in trigrams:
		if triplet not in tri_prob_wb:
			tri_prob_wb[triplet] = 0
		(temp1,temp2,temp3) = triplet
		pair = (temp2,temp3)
		tri_prob_wb[pair] = (float) (lamda_tri * tri_prob[triplet]) + (1-lamda_tri)*(bi_prob_wb[pair])
	return tri_prob_wb

tri_prob_wb = {}
tri_prob_wb = trigramprob_wb(trigrams,tri_prob,tri_prob_wb,bi_prob_wb)
#print tri_prob_wb

					##Plotting curves for Backoff
#For bigrams:

def getbigrams_wb(bigrams_wb,bigrams,bi_prob_wb,corpus):
	for sentence in corpus:
			for index, word in enumerate(sentence):
				if index > 0:
					pair = (sentence[index-1],word)
					if pair not in bigrams_wb:
						bigrams_wb[pair] = 0
					bigrams_wb[pair] = bi_prob_wb[pair] * tokens[sentence[index-1]]
	return bigrams_wb

bigrams_wb = {}
bigrams_wb = getbigrams_wb(bigrams_wb,bigrams,bi_prob_wb,corpus)
#print bigrams_wb

		#Plotting:
sorted_bigrams_wb = sorted(bigrams_wb.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_bigrams_wb,0,zipf)
#plotcurves(sorted_bigrams_wb,0,log)

#For trigrams:

def gettriigrams_wb(trigrams_wb,trigrams,tri_prob_wb,bigrams,corpus):
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in trigrams_wb:
					trigrams_wb[triplet] = 0
				pair = (sentence[index-2], sentence[index-1])
				trigrams_wb[triplet] = tri_prob_wb[triplet] * bigrams[pair]
	return trigrams_wb

trigrams_wb = {}
trigrams_wb = gettriigrams_wb(trigrams_wb,trigrams,tri_prob_wb,bigrams,corpus)
#print trigrams_wb

		#Plotting:
sorted_trigrams_wb = sorted(trigrams_wb.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_trigrams_wb,0,zipf)
#plotcurves(sorted_trigrams_wb,0,log)

						##Kneser-Ney smoothing:
		#For bigrams:

c_disc_bi = {}
for pair in bigrams:
	(temp1,temp2) = pair
	if temp1 not in c_disc_bi:
		c_disc_bi[temp1] = 0
	c_disc_bi[temp1] += 1
#print len(c_disc_bi)

def dval(pair,flag):
	if (flag == 0):
		d = 0.75
	elif (flag == 1):
		d = bi_prob_wb[pair]
	elif (flag == 2):
		d = bi_prob_ls[pair]
	return d

def lamdabi(word,word2,bigrams,tokens,flag):
	pair = (word,word2)
	return (dval(pair,flag)/(1.0*tokens[word]))*(c_disc_bi[word])
	
def P_cont(word,bigrams):
	return len(c_disc_bi)/(1.0*len(bigrams))

temp = tokens.keys()[0]
Na = P_cont(temp,bigrams)

def bigramprob_kn(bi_prob_kn,bigrams,tokens,flag):
	for pair in bigrams:
		(temp1,temp2) = pair
		if pair not in bi_prob_kn:
			bi_prob_kn[pair] = max(bigrams[pair] - dval(pair,flag), 0)/tokens[temp1] + lamdabi(temp1,temp2,bigrams,tokens,flag) * Na
	return bi_prob_kn

bi_prob_kn = {}
bi_prob_kn = bigramprob_kn(bi_prob_kn,bigrams,tokens,0)
bi_prob_kn_1 = {}
bi_prob_kn_1 = bigramprob_kn(bi_prob_kn_1,bigrams,tokens,1)
bi_prob_kn_2 = {}
bi_prob_kn_2 = bigramprob_kn(bi_prob_kn_2,bigrams,tokens,2)
#print 

		#For trigrams:

c_disc_tri = {}
for triplet in trigrams:
	(temp1,temp2,temp3) = triplet
	pair = (temp1,temp2)
	if pair not in c_disc_tri:
		c_disc_tri[pair] = 0
	c_disc_tri[pair] += 1

def lamdatri(pair,trigrams,bigrams):
	d = 0.75
	return (d/(1.0*bigrams[pair]))*(c_disc_tri[pair])

def trigramprob_kn(trigrams,bigrams,bi_prob_kn):
	d = 0.75
	for triplet in trigrams:
		(temp1,temp2,temp3) = triplet
		if triplet not in tri_prob_kn:
			tri_prob_kn[triplet] = max(trigrams[triplet]-d,0)/bigrams[(temp1,temp2)] + lamdatri((temp1,temp2),trigrams,bigrams) * bi_prob_kn[(temp2,temp3)]
	return tri_prob_kn

tri_prob_kn = {}
tri_prob_kn = trigramprob_kn(trigrams,bigrams,bi_prob_kn)
#print tri_prob_kn

						##Plotting curves for Kneser-ney:

		#For bigrams:

def getbigrams_kn(bigrams_kn,bigrams,bi_prob_kn,corpus):
	for sentence in corpus:
			for index, word in enumerate(sentence):
				if index > 0:
					pair = (sentence[index-1],word)
					if pair not in bigrams_kn:
						bigrams_kn[pair] = 0
					bigrams_kn[pair] = bi_prob_kn[pair] * tokens[sentence[index-1]]
	return bigrams_kn

bigrams_kn = {}
bigrams_kn = getbigrams_wb(bigrams_kn,bigrams,bi_prob_kn,corpus)
bigrams_kn_1 = {}
bigrams_kn_1 = getbigrams_wb(bigrams_kn_1,bigrams,bi_prob_kn_1,corpus)
bigrams_kn_2 = {}
bigrams_kn_2 = getbigrams_wb(bigrams_kn_2,bigrams,bi_prob_kn_2,corpus)
#print sorted_bigrams_kn

		#Plotting:
sorted_bigrams_kn = sorted(bigrams_kn.items(), key = lambda x: x[1], reverse=True)
sorted_bigrams_kn_1 = sorted(bigrams_kn_1.items(), key = lambda x: x[1], reverse=True)
sorted_bigrams_kn_2 = sorted(bigrams_kn_2.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_bigrams_kn,0,zipf)
#plotcurves(sorted_bigrams_kn,0,log)

		#For trigrams:

def gettriigrams_kn(trigrams_kn,trigrams,tri_prob_kn,bigrams,corpus):
	for sentence in corpus:
		for index, word in enumerate(sentence):
			if index > 1:
				triplet = (sentence[index-2], sentence[index-1], word)
				if triplet not in trigrams_kn:
					trigrams_kn[triplet] = 0
				pair = (sentence[index-2], sentence[index-1])
				trigrams_kn[triplet] = tri_prob_kn[triplet] * bigrams[pair]
	return trigrams_kn

trigrams_kn = {}
trigrams_kn = gettriigrams_kn(trigrams_kn,trigrams,tri_prob_kn,bigrams,corpus)
#print trigrams_kn

		#Plotting:
sorted_trigrams_kn = sorted(trigrams_kn.items(), key = lambda x: x[1], reverse=True)
#plotcurves(sorted_trigrams_kn,0,zipf)
#plotcurves(sorted_trigrams_kn,0,log)

						##Graphs:
				#Compare three smoothing techniques:
		#Bigrams:
	#Laplace:
#plotcurves(sorted_bigrams_ls,1,zipf)
	#Backoff:
#plotcurves(sorted_bigrams_wb,1,zipf)
	#Kneser-Ney:
#plotcurves(sorted_bigrams_kn,1,zipf)
#plt.show()
	#Laplace:
#plotcurves(sorted_bigrams_ls,1,log)
	#Backoff:
#plotcurves(sorted_bigrams_wb,1,log)
	#Kneser-Ney:
#plotcurves(sorted_bigrams_kn,1,log)
#plt.show()

		#Trigrams:		
	#Laplace:
#plotcurves(sorted_trigrams_ls,1,zipf)
	#Backoff:
#plotcurves(sorted_trigrams_wb,1,zipf)
	#Kneser-Ney:
#plotcurves(sorted_trigrams_kn,1,zipf)
#plt.show()
	#Laplace:
#plotcurves(sorted_trigrams_ls,1,log)
	#Backoff:
#plotcurves(sorted_trigrams_wb,1,log)
	#Kneser-Ney:
#plotcurves(sorted_trigrams_kn,1,log)
#plt.show()
				#Compare Pkns computed using Laplace and Witten-Bell estimates:

		#Bigrams:
#plotcurves(sorted_bigrams_kn,1,zipf)
#plotcurves(sorted_bigrams_kn_1,1,zipf)
#plotcurves(sorted_bigrams_kn_2,1,zipf)
#plt.show()

#plotcurves(sorted_bigrams_kn,1,log)
#plotcurves(sorted_bigrams_kn_1,1,log)
#plotcurves(sorted_bigrams_kn_2,1,log)
#plt.show()

		#Trigrams:

							##Generating text:

def cond(bigrams, key): 
    joint = {k[1] : v for k, v in bigrams.items() if k[0] == key}
    sum_count = sum(joint.values())
    return {k : v / float(sum_count) for k, v in joint.items() }

def generate(unigrams, bigrams, length, first_word = None):
    words = []
    if first_word == None:
        first_word = list(unigrams.keys())[random.randrange(0, len(unigrams))]
    words.append(first_word)
    for i in range(length - 1):
        prev = words[i]
        prev_dict = cond(bigrams, prev)
        next_word = sorted(prev_dict.items(), key = lambda x : x[1], reverse = True)[0]
        words.append(next_word[0])
    return words

gen_sen = generate(tokens, bigrams, 100)
print gen_sen