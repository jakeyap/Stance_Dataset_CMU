#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:46:51 2021
To plot the comments per facebook post collected
@author: jakeyap
"""

# DONE : collate data from old experiments
# DONE : run ablation on regression best hyper params
# DONE: check facebook posts num_comms vs num_rxns vs num_shares
# TODO3 : correlate 3 metrics of virality with regression fit
# TODO4 : make a model that feeds the stance labels back in
# TODO5 : run fed back model
# TODO6 : tokenize facebook posts
# TODO7 : make a model for facebook posts
# TODO8 : make a dataloader for facebook posts
# TODO9 : duplicate a mainfile for facebook posts

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fname_comms = './../Data/fb_data/2021-04-29_1517/filtered_comments.jsonl'
fname_posts = './../Data/fb_data/2021-04-29_1517/filtered_correlated_posts.jsonl'

data_posts = []
data_comms = []

with open(fname_comms) as f:    # open comments file
    lines = f.readlines()
    datalen = len(lines)
    for i in range(datalen):
        if i % 100000 == 0:
            print("Processing %d / %d comments" % (i, datalen))
        data_comms.append(json.loads(lines[i]))

with open(fname_posts) as f:    # open posts file
    lines = f.readlines()
    datalen = len(lines)
    for i in range(datalen):
        if i % 1000 == 0:
            print("Processing %d / %d posts" % (i, datalen))
        data_posts.append(json.loads(lines[i]))

df_posts = pd.DataFrame(data_posts)     # length 130K
df_comms = pd.DataFrame(data_comms)     # length 2.6M

counts_truth = df_posts.comment_count.to_numpy()
counts_mined = np.zeros(counts_truth.shape)

post_ids = [int(x) for x in df_posts.post_id]
post_ids = np.array(post_ids)

comm_post_ids = df_comms.post_id
comm_post_ids = np.array([int(x) for x in comm_post_ids])

counts_shares = df_posts.share_count.to_numpy()
counts_reacts = df_posts.reaction_count.to_numpy()

datalen = post_ids.shape[0]
for i in range(datalen):    # count actual comments collected per post
    post_id = post_ids[i]
    indexes = (comm_post_ids == post_id)
    num_com = indexes.sum()
    counts_mined[i] = num_com
    if i % 1000 == 0:
        print("Processing %d / %d posts" % (i, datalen))


EPSILON = 0.    
if True:    # for plotting scatter of true num comments vs actual collected
    plt.figure(1, figsize=(8,6))
    plt.scatter(x=counts_truth+EPSILON, 
                y=counts_mined+EPSILON, 
                s=2)
    plt.title('Facebook comments collected', size=15)
    plt.ylabel('Num comments found', size=14)
    plt.xlabel('Num comments according to metadata',size=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-1, 1e5])
    plt.xlim([1e-1, 1e5])
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linewidth=0.2)

if True:    # for plotting 2d histogram in log scale. Same as above but in 2D histogram form
    plt.figure(2, figsize=(8,6))
    max_val_truth = np.log10(counts_truth.max())    # get limits of log
    max_val_mined = np.log10(counts_mined.max())    # get limits of log
    
    bins_y = np.logspace(-1, max_val_mined, 50)     # for log histogram bins y axis
    bins_x = np.logspace(-1, max_val_truth, 50)     # for log histogram bins x axis
    
    plt.hist2d(x=counts_truth+EPSILON,
               y=counts_mined+EPSILON,
               bins=[bins_x, bins_y],
               norm=mpl.colors.LogNorm(), 
               cmap=mpl.cm.viridis)
    plt.colorbar()
    plt.title('Facebook comments collected', size=15)
    plt.ylabel('Num comments found', size=14)
    plt.xlabel('Num comments according to metadata',size=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-1, 1e5])
    plt.xlim([1e-1, 1e5])
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linewidth=0.2)
    
if True: 
    plt.figure(3, figsize=(8,6))
    plt.scatter(x=counts_shares+EPSILON,
                y=counts_reacts+EPSILON,
                s=2)
    plt.title('Facebook virality measures', size=15)
    plt.ylabel('Num Reactions', size=14)
    plt.xlabel('Num Shares',size=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-1, 1e5])
    plt.xlim([1e-1, 1e5])
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linewidth=0.2)
    
    plt.figure(4, figsize=(8,6))
    plt.scatter(x=counts_truth+EPSILON,
                y=counts_reacts+EPSILON,
                s=2)
    plt.title('Facebook virality measures', size=15)
    plt.ylabel('Num Reactions', size=14)
    plt.xlabel('Num Comments',size=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-1, 1e5])
    plt.xlim([1e-1, 1e5])
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linewidth=0.2)
    
    plt.figure(5, figsize=(8,6))
    plt.scatter(x=counts_shares+EPSILON,
                y=counts_truth+EPSILON,
                s=2)
    plt.title('Facebook virality measures', size=15)
    plt.ylabel('Num Comments', size=14)
    plt.xlabel('Num Shares',size=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-1, 1e5])
    plt.xlim([1e-1, 1e5])
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linewidth=0.2)
    