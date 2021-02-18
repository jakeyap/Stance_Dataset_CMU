#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:12:06 2021
Some utility functions to help with data processing
Adapted from tokenizer_v2, but has specific stuff to deal with pretraining data.
Doesnt remove hashtags and mentions. 
Split by topics ["General_Terms", "Iran_Deal", "Santa_Fe_Shooting", "Student_Marches"]
Split by response type ["Quote", "Reply"]
For pretraining data, no need to split by train/test sets. Just use all to learn twitter language model
@author: jakeyap
"""

import json
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import math        
import torch
import re
import tokenizer_v2 as tokenizer_helper
import process_event_universe as tweet_helper

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['[URL]'])

def get_tweet_id2text_dictionary(directory='./../Data/SRQ_Stance_Twitter/',
                                 fname='event_universe_dump_0.json'):
    ''' Gets the dictionary that maps tweetIDs to tweet text '''
    tweet_dict = None
    with open(directory+fname) as jsonfile:
        lines = jsonfile.read()
        tweet_dict = json.loads(lines)
    return tweet_dict

def get_pretraining_dataset():
    ''' 
    Gets the pretraining dataset, then fill in text data
    If text data is not present, drop the row
    '''
    # get the dataframe of headers first
    df = tweet_helper.open_event_universe_header(0)
    
    # get the dictionary that matches tweetIDs to text
    tweet_dict = get_tweet_id2text_dictionary()
    datalen = len(df)
    
    target_text = []
    response_text = []
    error_count = 0
    for i in range (datalen):
        if i % 10000 == 0:
            print('Filling in text %d' % i)
        id0 = df.target_id[i]
        id1 = df.response_id[i]
        try:
            text0 = tweet_dict[id0]
            target_text.append(text0)
        except KeyError:
            target_text.append(np.nan)
            error_count += 1
        try:
            text1 = tweet_dict[id1]
            response_text.append(text1)
        except KeyError:
            response_text.append(np.nan)
            error_count += 1
    print('Number of tweets without content : %d' % error_count)
    print('Removing errors')
    
    width = df.shape[1]
    df.insert(width+0,'target_text', target_text)
    df.insert(width+1,'response_text', response_text)
    df = df.dropna(axis=0)
    return df

def swap_words_in_sentence_pair(text0, text1):
    """
    Randomly chooses 1 word each in the text1 and text2, then flip them with adjacent word.
    Cannot be the same word. 
    example text : "i am feeling happy"
    swapped text : "i feeling am happy"

    Parameters
    ----------
    text0 : string
        first text.
    text1 : string
        second text.

    Returns
    -------
    text0 : string
        swapped version of text1.
    text1 : string
        swapped version of text2.
    swapped : binary int
        0 if original text, 1 if swapped
    """
    # TODO
    # convert strings to list of lists 1st
    list0 = text0.split(' ')    # split text into list of strings
    list1 = text1.split(' ')    # split text into list of strings
    
    if len(set(list0))==1 or len(set(list1))==1:
        print('Swap not valid for length 1')
        return text0, text1, 0
    
    while(True):
        length = len(list0) - 1
        idxA = np.random.randint(0, length)
        idxB = idxA + 1
        wordA = list0[idxA]
        wordB = list0[idxB]
        if wordA != wordB:
            list0[idxA] = wordB
            list0[idxB] = wordA
            break
    
    while(True):
        length = len(list1) - 1
        idxA = np.random.randint(0, length)
        idxB = idxA + 1
        wordA = list1[idxA]
        wordB = list1[idxB]
        if wordA != wordB:
            list1[idxA] = wordB
            list1[idxB] = wordA
            break
    text0 = ' '.join(list0)
    text1 = ' '.join(list1)
    return text0, text1, 1
        
def tokenize_and_encode_pretrain_df(dataframe,stopindex=1e9,max_length=128):    
    """
    Tokenize and encode the text into vectors, then stick inside dataframe
    Randomly select some of the samples to swap the words around. 

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe that contains all tweet data.
    stopindex : int, optional
        Number of tweets to stop at. The default is 1e9.

    Returns
    -------
    dataframe : pandas dataframe
        Original dataframe with additional information appended.

    """
    encoded_tweets = []
    token_type_ids = []
    attention_mask = []
    swapped = []
    
    counter = 0
    datalen = len(dataframe)
    
    swap_index = np.random.randint(0,2, size=(1,datalen))
    
    for i in range(datalen):
        try:
            text0 = dataframe.iloc[i]['clean_target_text']
            text1 = dataframe.iloc[i]['clean_response_text']
        except Exception:
            text0 = dataframe.iloc[i]['target_text']
            text1 = dataframe.iloc[i]['response_text']
        
        if swap_index[0, i]:    # if random index says swap, try swapping
            text0, text1, swap_sample = swap_words_in_sentence_pair(text0, text1)
        else:
            text0 = text0
            text1 = text1
            swap_sample = 0
        
        interaction = dataframe.iloc[i]['interaction_type']     # reply or quote
        topic = dataframe.iloc[i]['event']                      # get event
        topic = topic.replace('_', ' ')                         # replace underscore with space
        sep_token = ' [SEP] ' 
        text0 = interaction + sep_token + topic + sep_token + text0
        text1 = text1
        encoded_dict = tokenizer.__call__(text=text0,
                                          text_pair=text1,
                                          padding='max_length',
                                          truncation=True,
                                          is_split_into_words=False,
                                          max_length=max_length,
                                          return_tensors='pt')
        
        encoded_tweets.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        swapped.append(swap_sample)
        
        if counter % 100 == 0:
            print('Tokenizing comment: %00000d' % counter)
        if counter > stopindex:
            break
        counter = counter + 1
    
    width = dataframe.shape[1]
    dataframe.insert(width+0,'encoded_tweets', encoded_tweets)
    dataframe.insert(width+1,'token_type_ids', token_type_ids)
    dataframe.insert(width+2,'attention_mask', attention_mask)
    dataframe.insert(width+3,'swapped', swapped)
    return dataframe
    
if __name__ =='__main__':
    TOKENIZE = True
    MAXLENGTH = 256
    REMARK = 'test'
    ''' ========== Import data ========== '''
    df = get_pretraining_dataset()
    
    ''' ========== Convert into pandas ========== '''
    df_filtered, errors = tokenizer_helper.remove_nans(df)
    df_filtered = tokenizer_helper.clean_dataset(df_filtered) # remove excess spaces and URLs
    
    ''' ========== tokenize tweets, append to dataframe ========== '''
    if TOKENIZE:
        encoded_df = tokenize_and_encode_pretrain_df(dataframe=df_filtered, max_length=MAXLENGTH)
        torch.save(encoded_df, './../Data/SRQ_Stance_Twitter/event_universe_encoded_full_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    
    