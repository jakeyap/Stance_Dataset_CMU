#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:36:21 2021
This file is to handle the entire universe of tweets in the dataset
To prep the data for pretraining BERT to understand tweets

The goal is to make the model understand whether a pair is 'Quote' or 'Reply'
The goal is also to make the model figure out the topic
@author: jakeyap
"""
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from main_v2 import print_time

def pretraining_data_header():
    # opens dataset json header file returns them in a pandas dataframe
    
    time1 = time.time()
    DATADIR = './../Data/SRQ_Stance_Twitter/'
    FILENAME = 'event_universe.json'
    ''' ========== Import data ========== '''
    filename = DATADIR + FILENAME
    raw_list = []
    counter=0
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()
        for line in lines:
            thread_json = json.loads(line)
            raw_list.append(thread_json)
            if counter % 100 == 0:
                print('Importing json line: %0000d' % counter)
            counter = counter + 1
    print('Data length: %d' % len(raw_list))
    time2 = time.time()
    print_time(time1,time2)
    raw_df = pd.DataFrame(raw_list)
    return raw_df

def count_comments(dataframe):
    '''
    Counts number of pairs in dataset universe

    Parameters
    ----------
    dataframe : pandas dataframe
        pandas dataframe of dataset.

    Returns
    -------
    length : int
        length of dataset. w/o filtering, default is 4,691,512

    '''
    length = len(dataframe)
    print('Number of pairs : %d' % length)
    return length

# TODO 2 : count number of unique tweet IDs
def unique_tweets(dataframe):
    return

# TODO 3 : count number of quotes / replies
def count_interaction_type(dataframe):
    '''
    Counts number of quotes / replies

    Parameters
    ----------
    dataframe : pandas dataframe
        pandas dataframe of dataset.

    Returns
    -------
    num_quote : int
        number of quotes. w/o filtering, default is 3,825,367
    num_reply : int
        number of replies. w/o filtering, default is 866,145

    '''
    interaction = dataframe.interaction_type
    interaction = interaction.to_numpy()
    num_quote = (interaction=='Reply').sum()
    num_reply = (interaction=='Quote').sum()
    print('Quotes : %d' % num_quote)
    print('Replies: %d' % num_reply)
    return num_quote, num_reply
# TODO REACHED HERE
# TODO 4 : count number of topics
def count_topics(dataframe):
    '''
    Counts number of quotes / replies

    Parameters
    ----------
    dataframe : pandas dataframe
        pandas dataframe of dataset.

    Returns
    -------
    topics : list of strings
        list that contains topics. w/o filtering, defaults to ['General_Terms', 'Santa_Fe_Shooting', 'Iran_Deal', '']
    num_reply : int
        number of replies. w/o filtering, default is 866,145

    '''
    interaction = dataframe.interaction_type
    interaction = interaction.to_numpy()
    num_quote = (interaction=='Reply').sum()
    num_reply = (interaction=='Quote').sum()
    print('Quotes : %d' % num_quote)
    print('Replies: %d' % num_reply)
    return num_quote, num_reply

# TODO 5 : profile number of quotes/replies vs topics
    
# TODO 6 : set up twitter API via tweepy

# TODO 7 : download the raw tweet text of universe



if __name__ == '__main__':
    df = pretraining_data_header()