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
import tweepy
import os
from main_v2 import print_time

import argparse

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

def get_unique_tweet_ids(dataframe):
    ''' 
    count number of unique tweet IDs 
    Default number of unique source IDs:   2,949,486
    Default number of unique response IDs: 4,691,396
    Default number of unique tweets :      7,472,900
    returns the 3 sets
    '''
    
    source   = set(dataframe.target_id)     # set of source tweets
    response = set(dataframe.response_id)   # set of response tweets
    universe = source.union(response)       # set of all tweets
    
    len_source = len(source)
    len_response = len(response)
    len_universe = len(universe)
    
    print('Number of unique source tweets   %d' % len_source)
    print('Number of unique response tweets %d' % len_response)
    print('Number of unique tweets          %d' % len_universe)
    
    return source, response, universe

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

def count_topics(dataframe):
    '''
    Counts number of topics in ['General_Terms', 'Iran_Deal', 'Santa_Fe_Shooting', 'Student_Marches']
    
    Parameters
    ----------
    dataframe : pandas dataframe
        pandas dataframe of dataset.

    Returns
    -------
    num_general_terms : int
        number of General_Terms. w/o filtering, default is 4,582,406
    num_iran_deal : int
        number of Iran_Deal. w/o filtering, default is 41,152
    num_santa_fe : int
        number of Santa_Fe_Shooting. w/o filtering, default is 36,319
    num_students : int
        number of Student_Marches. w/o filtering, default is 31,635

    '''
    event = dataframe.event
    event = event.to_numpy()
    num_general_terms = (event=='General_Terms').sum()
    num_iran_deal = (event=='Iran_Deal').sum()
    num_santa_fe = (event=='Santa_Fe_Shooting').sum()
    num_students = (event=='Student_Marches').sum()
    print('General_Terms    : %d' % num_general_terms)
    print('Iran_Deal        : %d' % num_iran_deal)
    print('Santa_Fe_Shooting: %d' % num_santa_fe)
    print('Student_Marches  : %d' % num_students)
    return num_general_terms, num_iran_deal, num_santa_fe, num_students

# TODO 5 : profile number of quotes/replies vs topics

def get_twitter_keys():
    ''' get the twitter keys, return as dictionary '''
    fname = '~/.credentials_social_media/twitter1.key'  # key filename
    fname = os.path.expanduser(fname)                   # expand tilde
    file = open(fname, 'r')
    lines = []
    for line in file:
        lines.append(line.replace('\n', ''))
    
    key_dict = {'CONSUMER_KEY': lines[1],
                'CONSUMER_SECRET': lines[3],
                'ACCESS_TOKEN': lines[5],
                'ACCESS_TOKEN_SECRET': lines[7]}
    return key_dict

def get_twitter_api():
    ''' get tweepy to return a twitter api object '''
    key_dict = get_twitter_keys()
    CONSUMER_KEY = key_dict['CONSUMER_KEY']
    CONSUMER_SECRET = key_dict['CONSUMER_SECRET']
    ACCESS_TOKEN = key_dict['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = key_dict['ACCESS_TOKEN_SECRET']
    
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=120, retry_count=2, )
    return api

def get_tweets_bulk(api, tweet_ids_list):
    ''' Get a bunch of tweets in bulk from a list of numerical tweet IDs '''
    time1 = time.time()
    # loop thru list of list to extract buckets
    # store into dictionary
    
    tweet_dict = dict()
    list_of_lists = []
    counter = 0
    datalen = len(tweet_ids_list)
    # chop up the ID list into list of lists 
    # each list length is 100 cauz the api bucket is 100
    while counter < datalen:
        bucket = tweet_ids_list[counter : counter + 100]
        list_of_lists.append(bucket)
        counter += 100
    
    counter = 0
    for id_bucket in list_of_lists:
        try:
            bucket = api.statuses_lookup(id_bucket, tweet_mode='extended') # pass in 100 ids
            bucket_size = len(bucket)           # result might be shorter cauz of deleted tweets
            for i in range (bucket_size):       # within each bucket
                tweet = bucket[i]                   # find each tweet
                tweet_id = tweet.id                 # find each tweet ID
                tweet_text = tweet.full_text        # find each tweet text
                tweet_dict[tweet_id] = tweet_text   # add to dictionary
            counter += 1
            if counter % 10 == 0:
                print('Num of buckets done : %d' % counter, flush=True)
                time2 = time.time()
                print_time(time1, time2)
        except Exception as e:
            print(e)
    
    time3 = time.time()
    print_time(time1, time3)
    return tweet_dict

def get_dataset_tweets(api, dataframe, length=-1):
    tweet_ids = get_unique_tweet_ids(dataframe)
    universe_ids = list(tweet_ids[2])   # get all tweet IDs
    if length!=-1:
        universe_ids = universe_ids[:length]
    return get_tweets_bulk(api, universe_ids)

def get_tweets_single(api, tweet_id):
    ''' Get a single tweet from its numerical tweet ID'''
    return api.get_status(tweet_id, tweet_mode='extended')

def save_tweet_dict(tweet_dict,
                    directory = '~/Projects/Data/SRQ_Stance_Twitter/',
                    fname='event_universe_dump.json'):
    fname = directory + fname
    fname = os.path.expanduser(fname)   # expand tilde
    with open(fname, 'w') as file:
        file.write(json.dumps(tweet_dict, indent=0))

def open_event_universe_header(segment=-1):
    ''' 
    opens event_universe dataset json file 
    the file contains all tweet IDs returns them in a pandas dataframe 
    '''
    
    time1 = time.time()
    DATADIR = './../Data/SRQ_Stance_Twitter/'
    FILENAME = 'event_universe.json'
    ''' ========== Import data ========== '''
    filename = DATADIR + FILENAME
    raw_list = []
    counter=0
    
    with open(filename) as jsonfile:
        lines = jsonfile.readlines()
        if segment==-1:
            head_idx = 0
            tail_idx = len(lines)
        else:
            head_idx = segment * 200000
            tail_idx = (segment+1) * 200000
        for line in lines[ head_idx : tail_idx ]:
            thread_json = json.loads(line)
            raw_list.append(thread_json)
            if counter % 10000 == 0:
                print('Importing tweet pairs : %0000d' % counter, flush=True)
            counter = counter + 1
    print('Data length: %d' % len(raw_list), flush=True)
    time2 = time.time()
    print_time(time1,time2)
    raw_df = pd.DataFrame(raw_list)
    return raw_df


# TODO: get all the tweet data. attempting runs 1-19 now, should take 10hours plus to get
# TODO: need to merge them later
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', default=0, type=int, help='which segment of 200k tweets data to grab. range is [0-37] inclusive')
    args = parser.parse_args()
    segment = args.segment
    fname = 'event_universe_dump_'+str(segment)+'.json'
    df = open_event_universe_header(segment)
    print('Getting API', flush=True)
    api = get_twitter_api()
    print('Getting list of tweet IDs', flush=True)
    tweet_sets = get_unique_tweet_ids(df)
    all_tweet_ids = list(tweet_sets[2])
    print('Getting tweet text', flush=True)
    
    tweet_dict = get_tweets_bulk(api, all_tweet_ids)
    save_tweet_dict(tweet_dict, fname=fname)
    # this takes approx 60min to complete worst case