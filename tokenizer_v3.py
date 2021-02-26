#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:52:16 2021
Some utility functions to help with data processing
Version 3. Uses BerTweet tokenizer configuration
The examples are shown here
https://huggingface.co/vinai/bertweet-base

WARNING: the input token size for bertweet is 128 max. 

@author: jakeyap
"""
import os
import json
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import math        
import torch
import re

from tokenizer_v2 import empty_label_dictionary, convert_label_string2num, convert_label_num2string, remove_nans

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                          normalization=True)

def empty_label_dictionary(num_types=6):
    """
    Creates a dictionary of labels:counts

    Returns
    -------
    categories : dictionary
        Dictionary containing the counts of the labels.

    """
    if num_types==6:
        categories = {'Explicit_Denial':0,
                      'Implicit_Denial':0,
                      'Implicit_Support':0,
                      'Explicit_Support':0,
                      'Comment':0,
                      'Queries':0}
    elif num_types==4:
        categories = {'Denial':0,
                      'Support':0,
                      'Comment':0,
                      'Queries':0}    
    else:
        raise Exception
    return categories

   

def convert_interaction_type_string2num(interaction_type):
    '''
    Converts the interaction_type string labels into numbers

    Parameters
    ----------
    interaction_type : string
        Reply or Quote

    Returns
    -------
    int
        0 if reply, 1 if quote
    '''
    
    if interaction_type=='Reply':
        return 0
    else:
        return 1

def convert_interaction_type_num2string(number):
    '''
    Converts the interaction_type numbers into string labels

    Parameters
    ----------
    number : int
        number label of interaction_type.

    Returns
    -------
    str
        'Reply' is 0, 'Quote' if 1.

    '''
    if number==0:
        return 'Reply'
    else:
        return 'Quote'
        
def tokenize_and_encode_pandas(dataframe,stopindex=1e9,max_length=128):    
    """
    Tokenize and encode the text into vectors, then stick inside dataframe

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
    labels_6_types = []
    labels_4_types = []
    interaction_types = []
    
    counter = 0
    for i in range(len(dataframe)):
        try:
            #tokenized_parent= tokenizer.tokenize(dataframe.iloc[i]['clean_target_text'])
            #tokenized_tweet = tokenizer.tokenize(dataframe.iloc[i]['clean_response_text'])
            text_parent= dataframe.iloc[i]['clean_target_text']
            text_tweet = dataframe.iloc[i]['clean_response_text']
        except Exception:
            #tokenized_parent= tokenizer.tokenize(dataframe.iloc[i]['target_text'])
            #tokenized_tweet = tokenizer.tokenize(dataframe.iloc[i]['response_text'])
            text_parent= dataframe.iloc[i]['target_text']
            text_tweet = dataframe.iloc[i]['response_text']
        
        interaction = dataframe.iloc[i]['interaction_type']     # reply or quote
        topic = dataframe.iloc[i]['event']                      # get event
        topic = topic.replace('_', ' ')                         # replace underscore with space
        sep_token = ' [SEP] ' 
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
        start = 0
        lines = jsonfile.readlines()
        counter = 0                 # Variable to count the loop
        end = start + NUM_TO_IMPORT # End index
        for line in lines:
            if (counter >= start) and (counter < end):
                thread_json = json.loads(line)
                raw_list.append(thread_json)
            if (counter >= end):
                break
            if counter % 100 == 0:
                print('Importing json line: %0000d' % counter)
            counter = counter + 1
    
    ''' ========== Convert into pandas ========== '''
    pd_dataframe = pd.DataFrame(raw_list)
    return pd_dataframe


if __name__ =='__main__':
    NUM_TO_IMPORT = 1e9
    TOKENIZE = True
    TRAINING_RATIO = 0.8
    MAXLENGTH = 128
    
    
    DATADIR = './data/'
    FILENAME = 'stance_dataset.json'
    REMARK = 'bertweet'
    
    ''' ========== Import data into pandas========== '''    
    ''' ========== Convert into pandas ========== '''
    pd_dataframe = json_2_df(folder=DATADIR, fname=FILENAME)
    filtered_df, _= remove_nans(pd_dataframe)
    ''' ========== tokenize tweets, append to dataframe ========== '''
    if TOKENIZE:
        encoded_df = tokenize_and_encode_pandas(dataframe=filtered_df, max_length=MAXLENGTH)
    
    #Split by topics ["General Terms", "Iran_Deal", "Santa_Fe_Shooting", "Student Marches"]
    #Split by response type ["Quote", "Reply"]
    ''' ========== split into training and test sets ========== '''
    datalength = encoded_df.shape[0]
    train_index = round (TRAINING_RATIO * datalength)
    
    ''' ========== Shuffle and split dataframe rows ========== '''
    encoded_df = encoded_df.sample(frac=1)
    train_set = encoded_df.iloc[0:train_index].copy()
    test_set = encoded_df.iloc[train_index:].copy()
    
    ''' ========== save both datasets into binaries ========== '''
    torch.save(train_set, './data/train_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    torch.save(test_set, './data/test_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    """
    
    DATADIR = os.path.expanduser('~/Projects/Data/SemEval17/smu_processed/')
    FILENAME1 = 'semeval17_dev_flattened.json'
    FILENAME2 = 'semeval17_test_flattened.json'
    FILENAME3 = 'semeval17_train_flattened.json'
    REMARK = 'semeval17_bertweet'
    
    pd_dataframe1 = json_2_df(folder=DATADIR, fname=FILENAME1)
    filtered_df1,_= remove_nans(pd_dataframe1)
    pd_dataframe2 = json_2_df(folder=DATADIR, fname=FILENAME2)
    filtered_df2,_= remove_nans(pd_dataframe2)
    pd_dataframe3 = json_2_df(folder=DATADIR, fname=FILENAME3)
    filtered_df3,_= remove_nans(pd_dataframe3)
    
    if TOKENIZE:
        encoded_df1 = tokenize_and_encode_pandas(dataframe=filtered_df1, max_length=MAXLENGTH)
        encoded_df2 = tokenize_and_encode_pandas(dataframe=filtered_df2, max_length=MAXLENGTH)
        encoded_df3 = tokenize_and_encode_pandas(dataframe=filtered_df3, max_length=MAXLENGTH)
    
    ''' ========== combine train/dev sets into a big train set ========== '''
    train_set = pd.concat([encoded_df1, encoded_df3])
    test_set  = encoded_df2
    
    ''' ========== save both datasets into binaries ========== '''
    torch.save(train_set, './data/train_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    torch.save(test_set, './data/test_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    """