#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:59:47 2021
same as tokenizers_v4, except does things solo. not pairwise
@author: jakeyap
"""

import torch
import numpy as np
import json
import pandas as pd
import tweepy

import matplotlib.pyplot as plt

from transformers import AutoTokenizer

# from tokenizer_v2 import convert_label_string2num
from tokenizer_v3 import convert_interaction_type_string2num, convert_label_string2num

torch.manual_seed(0) # for fixing RNG seed

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                          normalization=True)



def tokenize_and_encode_pandas_pair(dataframe,stopindex=1e9,max_length=128, 
                                    bertweet=False, incl_meta=True):    
    """
    Tokenize and encode the text into vectors, then stick inside dataframe

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe that contains all tweet data.
    stopindex : int, optional
        Number of tweets to stop at. The default is 1e9.
    bertweet : bool, optional
        Whether to encode for bertweet. If for bertweet, the [SEP] token is different
    incl_meta : bool, optional
        Whether to include meta information about topic and interaction type in front
        
    Returns
    -------
    dataframe : pandas dataframe
        Original dataframe with additional information appended.

    """
    encoded_tweets = []
    token_type_ids = []
    attention_mask = []
    labels_6_types = []
    labels_4_types = []
    interaction_types = []
    
    counter = 0
    for i in range(len(dataframe)):
        text_parent= dataframe.iloc[i]['target_text']
        text_tweet = dataframe.iloc[i]['response_text']
        
        if incl_meta:
            interaction = dataframe.iloc[i]['interaction_type']     # reply or quote
            topic = dataframe.iloc[i]['event']                      # get event
            topic = topic.replace('_', ' ')                         # replace underscore with space
        else:
            interaction = ''
            topic = ''
        if bertweet:
            sep_token = ' </s> '  # tokenizer for bertweet is a bit different
        else:
            sep_token = ' [SEP] ' # tokenizer for regular bert uses [SEP] tokens
        text1 = interaction + sep_token + topic + sep_token + text_parent
        text2 = text_tweet
        encoded_dict = tokenizer.__call__(text=text1,
                                          text_pair=text2,
                                          padding='max_length',
                                          truncation=True,
                                          is_split_into_words=False,
                                          max_length=max_length,
                                          return_tensors='pt')
        
        encoded_tweets.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        
        label = dataframe.iloc[i]['label']
        labels_6_types.append(convert_label_string2num(label, num_types=6))
        labels_4_types.append(convert_label_string2num(label, num_types=4))
        
        interaction_type = dataframe.iloc[i]['interaction_type']
        interaction_types.append(convert_interaction_type_string2num(interaction_type))
        
        if counter % 100 == 0:
            print('Tokenizing comment: %00000d' % counter)
        if counter > stopindex:
            break
        counter = counter + 1
    
    # width = dataframe.shape[1]
    dataframe.insert(dataframe.shape[1],'encoded_tweets', encoded_tweets)
    dataframe.insert(dataframe.shape[1],'token_type_ids', token_type_ids)
    dataframe.insert(dataframe.shape[1],'attention_mask', attention_mask)
    try:
        dataframe.insert(dataframe.shape[1],'number_labels_6_types', labels_6_types)
        dataframe.insert(dataframe.shape[1],'number_labels_4_types', labels_4_types)
    except Exception as e:
        print(e)
    dataframe.insert(dataframe.shape[1],'interaction_type_num', interaction_types)
    return dataframe



def json_2_df(folder, fname):
    filename = folder+fname
    raw_list = []
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()
        counter = 0                 # Variable to count the loop
        for line in lines:
            thread_json = json.loads(line)
            raw_list.append(thread_json)
            if counter % 1000 == 0:
                print('Importing json line: %0000d' % counter)
            counter = counter + 1
    
    ''' ========== Convert into pandas ========== '''
    pd_dataframe = pd.DataFrame(raw_list)
    return pd_dataframe

if __name__ == '__main__':
    
    TRAINING_RATIO = 0.8
    
    folder1= './../Data/SRQ_Stance_Twitter/'
    fname1 = 'stance_dataset_tweet_dump.bin'
    
    tweet_dict = torch.load(folder1+fname1)
    
    head_tweets = []
    
    folder2= './data/'
    fname2 = 'stance_dataset.json'
    # dataframe original length 5221
    df = json_2_df(folder=folder2, fname=fname2) 
    
    # dataframe after drop nan length 4271
    df = df.dropna()
    
    print(len(df))
    tweet_ids = df.target_id
    retweets_counts = []
    favorite_counts= []
    error_index = []
    for i in range(len(tweet_ids)):
        tweet_id  = int(tweet_ids.iloc[i])
        try:
            json_data = tweet_dict[tweet_id]._json
            retweets_count = json_data['retweet_count']
            favorite_count= json_data['favorite_count']
            
            retweets_counts.append(retweets_count)
            favorite_counts.append(favorite_count)
        except KeyError:
            # some of the labelled tweets are not in the twitter data anymore. 
            # 813 errors to be exact. fill in with nans
            error_index.append(i)
            retweets_counts.append(np.nan)
            favorite_counts.append(np.nan)
    
    # slot in the new data wrt how many tweets
    df.insert(loc=df.shape[1], column='retweets_count', value=retweets_counts)
    df.insert(loc=df.shape[1], column='favorite_count', value=favorite_counts)
    
    # drop tweets with missing metadata. remaining data length 3458
    df = df.dropna()
    df = tokenize_and_encode_pandas_pair(df, max_length=128, bertweet=True, incl_meta=True)
    
    datalength0 = df.shape[0]
    train_index = round (TRAINING_RATIO * datalength0)
    
    ''' ========== Shuffle and split SRQ dataframe rows ========== '''
    df = df.sample(frac=1)
    train_set = df.iloc[0:train_index].copy() # length 2766
    test_set  = df.iloc[train_index:].copy() # length 692
    
    torch.save(train_set, './data/train_set_128_w_length_bertweet.bin')
    torch.save(test_set, './data/test_set_128_w_length_bertweet.bin')

    favorite_count_all = list(df.favorite_count)
    retweets_count_all = list(df.retweets_count)
    favorite_count_train = list(train_set.favorite_count)
    retweets_count_train = list(train_set.retweets_count)
    
    max_favorite = max(favorite_count_all)
    max_retweets = max(retweets_count_all)
    
    fig, axes = plt.subplots(1,2)
    ax0 = axes[0]
    ax1 = axes[1]
    log_bins = np.logspace(np.log10(0.1), np.log10(max_favorite), 20)
    
    ax0.set_title('Number of Likes', size=11)
    ax0.hist([favorite_count_all, favorite_count_train],
             bins=log_bins, 
             rwidth=0.8, color=['r','b'],label=['all','train_set'])
    ax0.set_xscale('log')
    ax0.legend()
    ax0.grid(True)
    
    ax1.set_title('Number of Retweets', size=11)
    ax1.hist([retweets_count_all, retweets_count_train],
             bins=log_bins, 
             rwidth=0.8, color=['r','b'],label=['all','train_set'])
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)
    fig.tight_layout()
