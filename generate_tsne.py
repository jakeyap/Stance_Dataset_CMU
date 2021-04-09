#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:45:44 2021

Generates the embedding of some words using BERTweet's tokenizer
Runs t-sne on embeddings, then generate t-sne plot

@author: jakeyap
"""
from misc_helpers import fmt_time_pretty
import time
import numpy as np
import pandas as pd
import torch 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
""" Import Bertweet, BERT """ 

""" Import the text """ 
fname20 = './data/user_top_20.bin'
fname15 = './data/user_top_15.bin'
fname10 = './data/user_top_10.bin'
fname05 = './data/user_top_5.bin'

file20 = torch.load(fname20)
file15 = torch.load(fname15)
file10 = torch.load(fname10)
file05 = torch.load(fname05)

time1 = time.time()
""" Encode the text using BERTweet, BERT """ 
tknizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                        normalization=True)
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
''' edit this to swap file '''
dictionary = file05
user_ids = list(dictionary.keys())
# np.random.shuffle(user_ids)
# user_ids = user_ids [0:400]
keywords = []

token_reprs = []
bert_reprs  = []

bertweet.cuda()
bertweet.eval()
# TODO 
# use bertweet's embedding layer later
embeddinglayer = bertweet.embeddings
for each_user in user_ids:
    keywords.append(dictionary[each_user])
    with torch.no_grad():
        user_words = dictionary[each_user]
        user_repr0 = tknizer.encode(user_words,         # shape=(1,128)
                                    max_length=128,
                                    truncation=True, 
                                    padding='max_length',
                                    return_tensors='pt')
        embeddinglayer = bertweet.embeddings
        user_repr1 = embeddinglayer(user_repr0.cuda())  # shape=(1,128,768)
        user_repr1 = user_repr1.cpu()[0,1:20,:]         # shape=(1,20,768). select only slots for keywords
        user_repr1 = user_repr1.sum(0, keepdim=True)    # shape=(1,768). sum across keywords
        #user_repr1 = bertweet(user_repr0.cuda())[1]
        #user_repr1 = user_repr1.to('cpu')
        token_reprs.append(user_repr0)
        bert_reprs.append(user_repr1)

""" Calculates t-sne """ 
token_reprs = torch.stack(token_reprs,0)    # shape=(n,1,128)
bert_reprs = torch.stack(bert_reprs,0)      # shape=(n,1,768)

token_reprs = token_reprs.squeeze(1)        # shape=(n,128)
bert_reprs = bert_reprs.squeeze(1)          # shape=(n,768)

df = pd.DataFrame()
df.insert(loc=df.shape[1], column='keywords', value=keywords)
df.insert(loc=df.shape[1], column='tsne_tokens', value=token_reprs)
df.insert(loc=df.shape[1], column='tsne_bertweet', value=bert_reprs)

PERPLEX_LIST = [100,]
PCA_DIM_LIST = [8,]

for PERPLEX in PERPLEX_LIST:
    for PCA_DIM in PCA_DIM_LIST:
        pca = PCA(n_components=PCA_DIM)
        pca_tokens = pca.fit_transform(token_reprs)
        pca_bertweet = pca.fit_transform(bert_reprs)
        
        tsne = TSNE(n_components=2, verbose=1, perplexity=PERPLEX, n_iter=500)
        
        #tsne_tokens = tsne.fit_transform(token_reprs)
        #tsne_bertweet = tsne.fit_transform(bert_reprs)
        
        #tsne_pca_tokens = tsne.fit_transform(pca_tokens)
        tsne_pca_bertweet = tsne.fit_transform(pca_bertweet)
        
        # df['tsne-bertweet-one'] = tsne_bertweet[:,0]
        # df['tsne-bertweet-two'] = tsne_bertweet[:,1]
        # df['tsne-tokens-one'] = tsne_tokens[:,0]
        # df['tsne-tokens-two'] = tsne_tokens[:,1]
        # df['tsne-pca-tokens-one'] = tsne_pca_tokens[:,0]
        # df['tsne-pca-tokens-two'] = tsne_pca_tokens[:,1]
        
        df['tsne-pca-bertweet-one'] = tsne_pca_bertweet[:,0]
        df['tsne-pca-bertweet-two'] = tsne_pca_bertweet[:,1]
        
        """ Plot t-sne results """ 
        fig, axes = plt.subplots(1,1)
        plt.suptitle('PERPLEXITY %d, PCA-DIM %d' %(PERPLEX, PCA_DIM))
        ax0 = axes
        
        #palette=sns.color_palette("hls", 10),
        sns.scatterplot(
            x="tsne-pca-bertweet-one", y="tsne-pca-bertweet-two",
            data=df,
            legend="full",
            alpha=0.3,
            ax=ax0
        )
        
time2 = time.time()
print(fmt_time_pretty(time2,time1))