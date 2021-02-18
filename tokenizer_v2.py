#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:19:41 2021
Some utility functions to help with data processing
Version 2. Doesnt remove hashtags and mentions. 
Split by topics ["General_Terms", "Iran_Deal", "Santa_Fe_Shooting", "Student_Marches"]
Split by response type ["Quote", "Reply"]
Split by train/test sets (90-10)
@author: jakeyap
"""

import json
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import math        
import torch
import re

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['[URL]'])

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

def convert_label_string2num(label, num_types):
    """
    Converts text label into a number
    
    Parameters
    ----------
    label : string
        Text label.
    num_types : int
        4 or 6 categories
        
    Returns
    -------
    Integer label
    """
    dictionary = empty_label_dictionary(num_types)
    all_labels = list(dictionary.keys())
    if num_types==4:
        label = label.replace('Implicit_', '')
        label = label.replace('Explicit_', '')
    return all_labels.index(label)
    
def convert_label_num2string(number, num_types):
    """
    Converts a numerical label back into a string

    Parameters
    ----------
    number : int
        Integer label.
    num_types : int
        4 or 6 categories
        
    Returns
    -------
    Text Label
    """
    dictionary = empty_label_dictionary(num_types)
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
    tweet_pair : string
        json format

    Returns
    -------
    None.

    """
    print(json.dumps(tweet_pair, indent=4))

def pandas_find_post_label_str(index, dataframe):
    """
    Returns the label of a tweet, in string form
    
    Parameters
    ----------
    index : int
        A row index.
    dataframe : pandas dataframe

    Returns
    -------
    label : string
        label of tweet in string form

    """
    return dataframe.at[index, 'label']

def pandas_find_post_label_num(index, dataframe):
    """
    Returns the label of a tweet, in integer form
    
    Parameters
    ----------
    index : int
        A row index.
    dataframe : pandas dataframe

    Returns
    -------
    label : int
        label of tweet in integer form

    """
    return dataframe.at[index, 'label_number']

def count_empty(dataframe):
    empty = 0
    for counter in range (len(dataframe)):
        # check parent and child tweets
        parent_tweet = dataframe.iloc[counter]['clean_target_text']
        child_tweet = dataframe.iloc[counter]['clean_response_text']
        
        if parent_tweet=='' or child_tweet=='':
            empty = empty+1
    return empty

def remove_nans(dataframe):
    """
    Removes the tweets with nans inside from dataframe

    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe that contains all the tweets.

    Returns
    -------
    dataframe : pandas dataframe object
        Pandas dataframe with filtered tweets.
    error_indices : list
        List of integers. Each int represents index of tweet with nans

    """
    # store the index where there's something wrong with the tweets
    error_indices = []
    
    # Find all the errors
    for counter in range (len(dataframe)):
        # check parent and child tweets
        parent_tweet = dataframe.iloc[counter]['target_text']
        #child_tweet = dataframe.iloc[counter]['response_text']
        isnan = False
        isempty = False
        try:
            isnan = math.isnan(parent_tweet)
        except Exception:
            #isempty = (parent_tweet == '') or (child_tweet == '')
            pass
        if isnan or isempty:
            error_indices.append(counter)
    
    return dataframe.drop(error_indices), error_indices

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
        
        '''
        encoded_dict = tokenizer.__call__(text=tokenized_parent,
                                          text_pair=tokenized_tweet,
                                          padding='max_length',
                                          truncation=True,
                                          is_split_into_words=True,
                                          max_length=max_length,
                                          return_tensors='pt')
        
        '''
        '''
        encoded_dict = tokenizer.encode_plus(text=tokenized_tweet,
                                             text_pair=tokenized_parent,
                                             max_length=max_length,
                                             pad_to_max_length=True)
        '''
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
    
    width = dataframe.shape[1]
    dataframe.insert(width+0,'encoded_tweets', encoded_tweets)
    dataframe.insert(width+1,'token_type_ids', token_type_ids)
    dataframe.insert(width+2,'attention_mask', attention_mask)
    dataframe.insert(width+3,'number_labels_6_types', labels_6_types)
    dataframe.insert(width+4,'number_labels_4_types', labels_4_types)
    dataframe.insert(width+5,'interaction_type_num', interaction_types)
    return dataframe


def remove_urls(tweet_in):
    '''
    Removes links that start with HTTP/HTTPS in the tweet

    Parameters
    ----------
    tweet_in : string
        tweet

    Returns
    -------
    tweet_out : string
        cleaned tweet.
    '''
    re_object = re.compile('http:\S*|https:\S*|www.\S*')
    tweet_out = re_object.sub(repl='[URL]', string=tweet_in)
    return tweet_out

def remove_spaces(string_in):
    '''
    Removes newlines and tabs
    Parameters
    ----------
    tweet_in : string
        tweet
    Returns
    -------
    tweet_out : string
        cleaned tweet.
    '''
    re_object = re.compile('\s')
    string_out = re_object.sub(repl=' ', string=string_in)
    return string_out

def clean_dataset(dataframe):
    # create a new column called clean_target_text and clean_response_text
    # process the tweets and stick into the columns
    # append number labels 
    
    clean_target_texts = []
    clean_response_texts = []
    
    for i in range(len(dataframe)):
        if i % 100 ==0:
            print('Cleaning dataframe: %d' % i)
        target_text = dataframe.iloc[i]['target_text']
        
        response_text = dataframe.iloc[i]['response_text']
        
        
        target_text = remove_urls(target_text)
        target_text = remove_spaces(target_text)
        response_text = remove_urls(response_text)
        response_text = remove_spaces(response_text)
        
        clean_target_texts.append(target_text)
        clean_response_texts.append(response_text)
        
    width = dataframe.shape[1]
    dataframe.insert(width+0,'clean_target_text', clean_target_texts)
    dataframe.insert(width+1,'clean_response_text', clean_response_texts)
    return dataframe

def explore_dataset(dataframe, index=None):
    # prints target and response text for each line in cleaned dataset
    if index is None:
        #randomly generate a number using torch's RNG
        mav_value = len(dataframe) 
        index = torch.randint(mav_value,(1,1)).item()
    print('Index ', index)
    print('\nOriginal target')
    print(dataframe.iloc[index]['target_text'])
    print('\nClean target')
    print(dataframe.iloc[index]['clean_target_text'])
    print('\nOriginal response')
    print(dataframe.iloc[index]['response_text'])
    print('\nClean response')
    print(dataframe.iloc[index]['clean_response_text'])
    return
    
if __name__ =='__main__':
    DATADIR = './data/'
    FILENAME = 'stance_dataset.json'
    REMARK = 'new'
    NUM_TO_IMPORT = 1e9
    TOKENIZE = True
    TRAINING_RATIO = 0.8
    MAXLENGTH = 256
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
                print('Importing json line: %0000d' % counter)
            counter = counter + 1
    
    ''' ========== Convert into pandas ========== '''
    pd_dataframe = pd.DataFrame(raw_list)
    df_filtered, errors = remove_nans(pd_dataframe)
    df_filtered = clean_dataset(df_filtered)
    
    # split by quote or reply
    ''' ========== Plot the label densities ========== '''
    '''
    count1 = empty_label_dictionary()
    count2 = empty_label_dictionary()
    label_list = count1.keys()
    
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
    plt.xticks(xpts, label_list)
    plt.ylabel('Counts')
    plt.xlabel('Labels')
    plt.title('CMU twitter dataset labels')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    '''
    ''' ========== tokenize tweets, append to dataframe ========== '''
    if TOKENIZE:
        encoded_df = tokenize_and_encode_pandas(dataframe=df_filtered, max_length=MAXLENGTH)
    
    
    #Split by topics ["General Terms", "Iran_Deal", "Santa_Fe_Shooting", "Student Marches"]
    #Split by response type ["Quote", "Reply"]
    ''' ========== split into training and test sets ========== '''
    datalength = encoded_df.shape[0]
    train_index = round (TRAINING_RATIO * datalength)
    
    ''' ========== Shuffle and split dataframe rows ========== '''
    encoded_df = encoded_df.sample(frac=1)
    train_set = encoded_df.iloc[0:train_index].copy()
    test_set = encoded_df.iloc[train_index:].copy()
    
    ''' ========== count the labels for both sets ========== '''
    '''
    count3 = empty_label_dictionary()
    count4 = empty_label_dictionary()
    
    # Count number of labels
    datalength3 = train_set.shape[0]
    datalength4 = test_set.shape[0]
    # Go through training data to count labels
    for row in range(datalength3):
        string_label = train_set.iloc[row]['label']
        count3[string_label] = count3[string_label] + 1
    
    # Go through test data to count labels
    for row in range(datalength4):
        string_label = df_filtered.iloc[row]['label']
        count4[string_label] = count4[string_label] + 1 
    
    # Normalize both histograms
    train_label_max = sum(count3.values())
    test_label_max = sum(count4.values())
    for each_key in count3.keys():
        count3[each_key] = count3[each_key] * 100 / train_label_max
        count4[each_key] = count4[each_key] * 100 / test_label_max
        
    plt.figure(2)
    plt.bar(x=xpts-width/2,height=count3.values(), width=width, label='train-3844')
    plt.bar(x=xpts+width/2,height=count4.values(), width=width, label='test-427')
    plt.xticks(xpts, label_list)
    plt.ylabel('Counts')
    plt.xlabel('Labels')
    plt.title('CMU twitter dataset labels %')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    '''
    ''' ========== save both datasets into binaries ========== '''
    torch.save(train_set, './data/train_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
    torch.save(test_set, './data/test_set_'+str(MAXLENGTH)+'_'+REMARK+'.bin')
