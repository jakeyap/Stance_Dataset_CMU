# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:11:50 2020

@author: Yong Keong
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import re 
import tokenizer_utilities

NUMBER=5

#train_set = df.iloc[0:train_index].copy() # length 2752
#test_set  = df.iloc[train_index:].copy() # length 688

train_set = torch.load('./data/train_set_128_individual_bertweet_keywords_'+str(NUMBER)+'.bin')
test_set = torch.load('./data/test_set_128_individual_bertweet_keywords_'+str(NUMBER)+'.bin')

df = pd.concat([train_set, test_set])

favorite_count_all = df.favorite_count.to_numpy()
favorite_count_train = train_set.favorite_count.to_numpy()
favorite_count_test = test_set.favorite_count.to_numpy()

# rescale by weightage
favorite_count_train = favorite_count_train
favorite_count_test = favorite_count_test

labels_all = list(df.number_labels_4_types)
labels_train = list(train_set.number_labels_4_types)
labels_test = list(test_set.number_labels_4_types)
labels_str = ['Deny','Support','Comment','Query']

fig, axes = plt.subplots(1,2)
ax0 = axes[0]
ax1 = axes[1]

''' plot profile for number of likes '''
max_favorite = max(favorite_count_all)                              # find max of log scale
log_bins = np.logspace(np.log10(0.1), np.log10(max_favorite), 20)   # make log scale

ax0.set_title('Number of Likes', size=11)
ax0.hist([favorite_count_all, 
          favorite_count_train, 
          favorite_count_test],
         bins=log_bins, 
         rwidth=0.8, color=['r','b','g'],label=['all','train_set','test_set'])
ax0.set_xscale('log')
ax0.legend()
ax0.grid(True)

''' plot profile for label types '''
ax1.set_title('Label types', size=11)
ax1.hist([labels_all, 
          labels_train,
          labels_test],
         bins=[-0.5,0.5,1.5,2.5,3.5],
         rwidth=0.5, color=['r','b','g'],label=['all','train_set','test_set'])

ax1.set_xticks([0., 1., 2., 3.])
ax1.set_xticklabels(labels_str)
# bins=log_bins, 
#ax1.set_xscale('log')
ax1.legend()
ax1.grid(True)
fig.tight_layout()