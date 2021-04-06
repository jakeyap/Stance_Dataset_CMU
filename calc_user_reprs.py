#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:06:35 2021
To calculate user representations

shows 2 examples of TFIDF calculations. 
method 1 is from scratch
method 2 is just use API

@author: jakeyap
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from misc_helpers import fmt_time_pretty

import time

detokenizer = TreebankWordDetokenizer()

""" ==================== METHOD 1: FROM SCRATCH ==================== """
"""
tweetA = 'the man went out for a walk'
tweetB = 'the children sat around the fire'

bagOfWordsA = tweetA.split(' ')     # split into words
bagOfWordsB = tweetB.split(' ')     # split into words

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))  # obtain dictionary of words

numOfWordsA = dict.fromkeys(uniqueWords, 0) # for storing word counts
for word in bagOfWordsA:                    # go thru sentence
    numOfWordsA[word] += 1                  # count number of words 
numOfWordsB = dict.fromkeys(uniqueWords, 0) # for storing word counts
for word in bagOfWordsB:                    # go thru sentence
    numOfWordsB[word] += 1                  # count number of words 
    
stopwords.words('english')  # remove english stopwords

def computeTF(wordDict, bagOfWords):
    '''
    Calculates Term Frequencies as fraction of total count

    Parameters
    ----------
    wordDict : dictionary. key=words, value=counts
        Counts of words.
    bagOfWords : TYPE
        DESCRIPTION.

    Returns
    -------
    tfDict : TYPE
        DESCRIPTION.

    '''
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
idfs = computeIDF([numOfWordsA, numOfWordsB])

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df1 = pd.DataFrame([tfidfA, tfidfB])
"""

""" ==================== METHOD 2: JUST USE API ==================== """
"""
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([tweetA, tweetB])    # sparse matrix
feature_names = vectorizer.get_feature_names()          # get the words
dense = vectors.todense()                               # convert to normal matrix
denselist = dense.tolist()
df2 = pd.DataFrame(denselist, columns=feature_names)    # different from manual version cauz of internal smoothing functions
"""

import torch
from transformers import AutoTokenizer
import os

file = '~/Projects/Data/SRQ_Stance_Twitter/users_timeline_tweets.bin'
file = os.path.expanduser(file)

time1 = time.time()
vectorizer = TfidfVectorizer()

data = torch.load(file)
user_ids = list(data.keys())

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                          normalization=True)

def clean_special_tokens(string):
    string = string.replace('<s>', '')
    string = string.replace('</s>', '')
    string = string.replace('HTTPURL', '')
    string = string.replace('@USER', '')
    return string

def remove_stopwords(text, s_words):
    '''
    Removes stopwords from tokenized text

    Parameters
    ----------
    text : string
        duh
    s_words : list of strings
        each element is a stopword.

    Returns
    -------
    filtered : TYPE
        DESCRIPTION.

    '''
    filtered = []
    tokens = word_tokenize(text)                # this is a full tokenization. some stopwords get chopped also, e.g. shouldn't
    restored = detokenizer.detokenize(tokens)   # sentence has space between every word, punctuation. stopwords are properly restored
    words = restored.split()                    # split by spaces now into actual words rather than tokens
    for word in words:
        if word not in s_words:
            filtered.append(word)
    filtered = ' '.join(filtered)
    return filtered
    
s_words = set(stopwords.words('english'))  # get stopwords

user_summaries = []
counter = 0
for each_id in user_ids:
    tweets = data[each_id] # list of tweets
    user_summary = ""
    for each_tweet in tweets:
        lower = each_tweet.lower()          # convert to lower case
        encoded = tokenizer.encode(lower)   # encode to find URLs, @mentions
        text = tokenizer.decode(encoded)    # undo to get back into text
        text = clean_special_tokens(text)   # remove URLs, @mentions, <s> </s> tags
        text = remove_stopwords(text,       # remove stopwords
                                s_words)
        user_summary += text + ' '
    user_summaries.append(user_summary)
    counter += 1
    if counter % 200 == 0:
        print(counter, end=' ', flush=True)
        time2 = time.time()
        print(fmt_time_pretty(time1, time2))
    

time2 = time.time()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(user_summaries)      # returns a sparse matrix
word_names = vectorizer.get_feature_names()             # get the words
word_names = np.array(word_names)
datalen = vectors.shape[0]                              # get number of users

""" 
# AVOID THIS. IT BLOWS UP THE MEMORY DURING EXPANSION
dense = vectors.todense()                               # convert to normal matrix
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)     # different from manual version cauz of internal smoothing functions
"""
TOP_RANK = 20
user_reprs = dict()

for i in range(datalen):
    tfidf_vals = vectors[i,:].toarray()     # convert to normal array
    tfidf_vals = tfidf_vals.reshape(-1)     # reshape into shape (n,)
    tfidf_idx = tfidf_vals.argsort()        # get sorted index. low to high
    
    idx = tfidf_idx[-TOP_RANK:]             # get top ranked indices
    topwords = word_names[idx]              # get the top words
    topwords = topwords.tolist()            # convert to list
    topwords = ', '.join(topwords)          # stick together as a phrase
    user_id = user_ids[i]                   # find user ID for this user
    user_reprs[user_id] = topwords          # store into dictionary
    if i % 200 == 0:
        print(counter, end=' ', flush=True)
        time2 = time.time()
        print(fmt_time_pretty(time1, time2))

fname = './data/user_top_'+str(TOP_RANK)+'.bin' # filename to store
torch.save(user_reprs, fname)
