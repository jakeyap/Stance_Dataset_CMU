#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:16:05 2020
Some utility functions to help with data processing
@author: jakeyap
"""
import json
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import math        

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def empty_label_dictionary():
    """
    Creates a dictionary of labels:counts

    Returns
    -------
    categories : dictionary
        Dictionary containing the counts of the labels.

    """
    categories = {'Explicit_Denial':0,
                  'Implicit_Denial':0,
                  'Implicit_Support':0,
                  'Explicit_Support':0,
                  'Comment':0,
                  'Queries':0}
    return categories

def convert_label_string2num(label):
    """
    Converts text label into a number
    
    Parameters
    ----------
    label : string
        Text label.

    Returns
    -------
    Integer label
    """
    dictionary = empty_label_dictionary()
    all_labels = list(dictionary.keys())
    return all_labels.index(label)
    
def convert_label_num2string(number):
    """
    Converts a numerical label back into a string

    Parameters
    ----------
    number : int
        Integer label.

    Returns
    -------
    Text Label
    """
    dictionary = empty_label_dictionary()
    all_labels = list(dictionary.keys())
    return all_labels[number]


def print_json_file(json_filename, start=0, count=5, debug=False):
    """
    Pretty prints a few samples inside the json file

    Parameters
    ----------
    json_filename : string
        text of database file name.
    start : int, optional
        Index to start printing from. The default is 0.
    count : TYPE, optional
        How many items to print. The default is 5.
    debug : Boolean, optional
        True if need to save the logfile. The default is False.

    Returns
    -------
    None.

    """
    import json
    if debug:
        logfile = open('logfile.txt', 'w')
    with open(json_filename) as jsonfile:
        counter = 0
        lines = jsonfile.readlines()
        counter = 0         # Variable to count the loop
        end = start+count   # End index
        for line in lines:
            if (counter >= start) and (counter < end):
                reader = json.loads(line) 
                # If this thread has posts, enter the thread
                # the dumps function helps to display json data nicely
                helper = json.dumps(reader, indent=4)
                print(helper)
                if debug:
                    logfile.write(helper)
            if counter >= end:
                break
            counter = counter + 1
    if debug:
        logfile.close()

def print_json_tweet_pair(tweet_pair):
    """
    Pretty prints a tweet

    Parameters
    ----------
    comment : string

    Returns
    -------
    None.

    """
    print(json.dumps(tweet_pair, indent=4))

def pandas_find_post_label_str(index, dataframe):
    return dataframe.at[index, 'label']

def pandas_find_post_label_num(index, dataframe):
    return dataframe.at[index, 'label_number']

def remove_nans(dataframe):
    
    # store the index where there's something wrong with the tweets
    error_indices = []
    
    # Find all the errors
    for counter in range (len(dataframe)):
        # check parent and child tweets
        parent_tweet = dataframe.iloc[counter]['target_text']
        child_tweet = dataframe.iloc[counter]['response_text']
        isnan = False
        isempty = False
        try:
            isnan = (math.isnan(parent_tweet)) or (math.isnan(child_tweet))
        except Exception:
            isempty = (parent_tweet == '') or (child_tweet == '')
        if isnan or isempty:
            error_indices.append(counter)
        counter = counter + 1
    
    return dataframe.drop(error_indices), error_indices

def tokenize_and_encode_pandas(dataframe,stopindex=1e9):    
    encoded_tweets = []
    token_type_ids = []
    attention_mask = []
    labels = []
    counter = 0
    for i in range(len(dataframe)):
        tokenized_parent= tokenizer.tokenize(dataframe.iloc[i]['target_text'])
        tokenized_tweet = tokenizer.tokenize(dataframe.iloc[i]['response_text'])
        encoded_dict = tokenizer.encode_plus(text=tokenized_tweet,
                                             text_pair=tokenized_parent,
                                             max_length=128,
                                             pad_to_max_length=True)
        encoded_tweets.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        
        label = dataframe.iloc[i]['label']
        labels.append(convert_label_string2num(label))
                
        if counter % 100 == 0:
            print('Tokenizing comment: %00000d' % counter)
        if counter > stopindex:
            break
        counter = counter + 1
    
    width = dataframe.shape[1]
    dataframe.insert(width+0,'encoded_tweets', encoded_tweets)
    dataframe.insert(width+1,'token_type_ids', token_type_ids)
    dataframe.insert(width+2,'attention_mask', attention_mask)
    dataframe.insert(width+3,'number_labels', labels)
    return dataframe

if __name__ =='__main__':
    DATADIR = './data/'
    FILENAME = 'stance_dataset.json'
    NUM_TO_IMPORT = 1e9
    TOKENIZE = True
    
    ''' ========== Import data ========== '''
    filename = DATADIR + FILENAME
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
                print('Flattening thread to pair: %0000d' % counter)
            counter = counter + 1
    
    ''' ========== Convert into pandas ========== '''
    pd_dataframe = pd.DataFrame(raw_list)
    df_filtered, errors = remove_nans(pd_dataframe)
    
    ''' ========== Plot the label densities ========== '''
    count1 = empty_label_dictionary()
    count2 = empty_label_dictionary()
    
    # Count number of labels
    datalength1 = pd_dataframe.shape[0]
    datalength2 = df_filtered.shape[0]
    # Go through dataset to count labels
    for row in range(datalength1):
        string_label = pd_dataframe.iloc[row]['label']
        count1[string_label] = count1[string_label] + 1
    
    # Go through filtered dataset to count labels
    for row in range(datalength2):
        string_label = df_filtered.iloc[row]['label']
        count2[string_label] = count2[string_label] + 1
        
    # Actual plotting
    import matplotlib.pyplot as plt
    plt.figure(1)
    xpts = np.arange(len(count1))
    width = 0.25
    plt.bar(x=xpts-width/2,height=count1.values(), width=width, label='raw')
    plt.bar(x=xpts+width/2,height=count2.values(), width=width, label='filtered')
    plt.ylabel('Counts')
    plt.xlabel('Labels')
    plt.title('CMU twitter dataset labels')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    ''' ========== tokenize tweets, append to dataframe ========== '''
    if TOKENIZE:
        encoded_df = tokenize_and_encode_pandas(dataframe=df_filtered)
    
    ''' ========== split into training and test sets ========== '''
    ''' ========== plot the label densities for both ========== '''
    ''' ========== save both datasets into binaries ========== '''
    