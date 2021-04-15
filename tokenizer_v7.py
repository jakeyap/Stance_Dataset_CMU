#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:47:53 2021

For hierarchical bertweet. major changes from tokenizers_v6
    dont use attention mask and token type IDs. 
    includes bert tokenizer's output on the same txt
    stick topic + username into front of each tweet text    
    encode tweets individually rather than pairwise
    includes root post's username and user_followers into dataframe
    includes tail post's username and user_followers into dataframe
    includes keywords to represent user (user_keywords) into dataframe
@author: jakeyap
"""

import torch
import numpy as np
import json
import pandas as pd
import tweepy

import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers import BertTokenizer

# from tokenizer_v2 import convert_label_string2num
from tokenizer_v3 import convert_interaction_type_string2num, convert_label_string2num

torch.manual_seed(0) # for fixing RNG seed

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",
                                          normalization=True)
bert_tkzer = BertTokenizer.from_pretrained("bert-base-uncased")
   

def tokenize_and_encode_pandas_pair(dataframe,stopindex=1e9,max_length=128):    
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
    encodings_head = []
    encodings_tail = []
    user_keywords_enc = []
    
    encodings_head_bert = []
    encodings_tail_bert = []
    user_keywords_enc_bert = []
    
    labels_6_types = []
    labels_4_types = []
    interaction_types = []
    
    sep_token = ' </s> '  # tokenizer for bertweet is a bit different
    
    counter = 0
    for i in range(len(dataframe)):
        topic = dataframe.iloc[i]['event']              # get event
        topic = topic.replace('_', ' ')                 # replace underscore with space
        dataframe.iloc[i]['event'] = topic              # spell event in dataframe correctly
        
        head_user= dataframe.iloc[i]['usernames_head']  # get head's tweet ID
        tail_user= dataframe.iloc[i]['usernames_tail']  # get tail's tweet ID
        
        head_txt = dataframe.iloc[i]['target_text']     # get head's tweet text
        tail_txt = dataframe.iloc[i]['response_text']   # get tail's tweet text
        
        head_txt = head_user + sep_token + head_txt     # insert username to head
        tail_txt = tail_user + sep_token + tail_txt     # insert username to tail
        head_txt = topic + sep_token + head_txt         # insert topic to head
        tail_txt = topic + sep_token + tail_txt         # insert topic to tail
        
        keywords = dataframe.iloc[i]['users_keywords']  # get the user keywords representation
        
        head_encoding = tokenizer.encode(text=head_txt,       # for root post
                                         padding='max_length',
                                         truncation=True,
                                         is_split_into_words=False,
                                         max_length=max_length,
                                         return_tensors='pt')
        
        tail_encoding = tokenizer.encode(text=tail_txt,       # for tail post
                                         padding='max_length',
                                         truncation=True,
                                         is_split_into_words=False,
                                         max_length=max_length,
                                         return_tensors='pt')
        keywords_encoding = tokenizer.encode(keywords,              # shape=(1,128))
                                             padding='max_length',
                                             truncation=True,
                                             is_split_into_words=False,
                                             max_length=max_length,
                                             return_tensors='pt')
        
        ''' for converting bertweet encodings back into bert encodings '''
        txt_decode_h = tokenizer.decode(head_encoding.reshape(-1))          # convert head back into text
        txt_decode_t = tokenizer.decode(tail_encoding.reshape(-1))          # convert tail back into text
        keywords_decode = tokenizer.decode(keywords_encoding.reshape(-1))   # convert 
        txt_decode_h = txt_decode_h.replace('<s>','[CLS]').replace('<pad>','[PAD]').replace('</s>','[SEP]')
        txt_decode_t = txt_decode_t.replace('<s>','[CLS]').replace('<pad>','[PAD]').replace('</s>','[SEP]')
        keywords_decode = keywords_decode.replace('<s>','[CLS]').replace('<pad>','[PAD]').replace('</s>','[SEP]')
        
        head_encoding_bert = bert_tkzer.encode(text=txt_decode_h,           # bert encoding for head post
                                               padding='max_length',
                                               truncation=True,
                                               is_split_into_words=False,
                                               max_length=max_length,
                                               return_tensors='pt')
        tail_encoding_bert = bert_tkzer.encode(text=txt_decode_t,           # bert encoding for tail post
                                               padding='max_length',
                                               truncation=True,
                                               is_split_into_words=False,
                                               max_length=max_length,
                                               return_tensors='pt')
        keywords_encoding_bert = bert_tkzer.encode(text=keywords_decode,    # bert encoding for keywords
                                                   padding='max_length',
                                                   truncation=True,
                                                   is_split_into_words=False,
                                                   max_length=max_length,
                                                   return_tensors='pt')
        
        ''' append all the bertweet and bert tokenizer encodings into lists '''
        encodings_head.append(head_encoding)
        encodings_tail.append(tail_encoding)
        user_keywords_enc.append(keywords_encoding)
        encodings_head_bert.append(head_encoding_bert)
        encodings_tail_bert.append(tail_encoding_bert)
        user_keywords_enc_bert.append(keywords_encoding_bert)
        
        label = dataframe.iloc[i]['label']
        labels_6_types.append(convert_label_string2num(label, num_types=6))
        labels_4_types.append(convert_label_string2num(label, num_types=4))
        
        # includes root post's username, user_followers into dataframe
        interaction_type = dataframe.iloc[i]['interaction_type']
        interaction_types.append(convert_interaction_type_string2num(interaction_type))
        
        if counter % 100 == 0:
            print('Tokenizing comment: %00000d' % counter)
        if counter > stopindex:
            break
        counter = counter + 1
    
    # width = dataframe.shape[1]
    dataframe.insert(dataframe.shape[1],'encoded_tweets_h', encodings_head)
    dataframe.insert(dataframe.shape[1],'encoded_tweets_t', encodings_tail)
    dataframe.insert(dataframe.shape[1],'encoded_keywords', user_keywords_enc)
    dataframe.insert(dataframe.shape[1],'encoded_tweets_h_bert', encodings_head_bert)
    dataframe.insert(dataframe.shape[1],'encoded_tweets_t_bert', encodings_tail_bert)
    dataframe.insert(dataframe.shape[1],'encoded_keywords_bert', user_keywords_enc_bert)
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
    
    folder1= './../Data/SRQ_Stance_Twitter/'    # for dataset's full set of tweets
    fname1 = 'stance_dataset_tweet_dump.bin'    # for dataset's full set of tweets
    
    data_obj = torch.load(folder1+fname1)
    tweet_dict = data_obj['tweet_dict']
    user_dict = data_obj['user_dict']
    
    folder2= './data/'                          # tweets with annotated stance 
    fname2 = 'stance_dataset.json'              # tweets with annotated stance 
    # dataframe original length 5221
    df = json_2_df(folder=folder2, fname=fname2) 
    
    # dataframe after drop nan length 4271
    df = df.dropna()
    print(len(df))
    NUMBER = 20 # can be 5, 10, 15, 20
    fname3 = './data/user_top_'+str(NUMBER)+'.bin'
    user_keywords_dict = torch.load(fname3)
    
    head_ids = df.target_id
    tail_ids = df.response_id
    retweets_counts = []
    favorite_counts = []
    usernames_head  = []
    usernames_tail  = []
    followers_head  = []
    followers_tail  = []
    users_keywords  = []
    
    error_index = []
    for i in range(len(head_ids)):
        head_id  = int(head_ids.iloc[i])
        try:
            json_data = tweet_dict[head_id]._json           # get tweet's JSON
            retweets_count = json_data['retweet_count']     # get retweet count
            favorite_count = json_data['favorite_count']    # get likes count
            json_user = json_data['user']                   # get user object from tweet
            user_id = json_user['id']                       # get user ID from user obj
            user = user_dict[user_id]                       # get full user obj from user database
            username = user.name                            # get user name
            followers_count = user.followers_count          # get user follower count
            user_keywords = user_keywords_dict[user_id]     # get the keywords to represent user
            
            retweets_counts.append(retweets_count)          # store retweet count
            favorite_counts.append(favorite_count)          # store likes count
            usernames_head.append(username)                 # store username string
            followers_head.append(followers_count)          # store follower count
            users_keywords.append(user_keywords)            # store user representation
        except KeyError:
            # some of the labelled tweets are not in the twitter data anymore. 
            # 813 errors to be exact. fill in with nans
            error_index.append(i)
            retweets_counts.append(np.nan)
            favorite_counts.append(np.nan)
            usernames_head.append(np.nan)
            followers_head.append(np.nan)
            users_keywords.append(np.nan)
        
    print(len(error_index))
    error_index = []
    for i in range(len(tail_ids)):
        tail_id  = int(tail_ids.iloc[i])
        try:
            json_data = tweet_dict[tail_id]._json
            json_user = json_data['user']
            user_id = json_user['id']
            user = user_dict[user_id]
            username = user.name
            followers_count = user.followers_count
            
            usernames_tail.append(username)
            followers_tail.append(followers_count)
        except KeyError:
            # some of the labelled tweets are not in the twitter data anymore. 
            # for replies / quotes, it is ok, just append some filler data
            error_index.append(i)
            usernames_tail.append('unknown user')
            followers_tail.append(0)
    print(len(error_index))
    # slot in the new data wrt how many tweets
    df.insert(loc=df.shape[1], column='retweets_count', value=retweets_counts)
    df.insert(loc=df.shape[1], column='favorite_count', value=favorite_counts)
    df.insert(loc=df.shape[1], column='usernames_head', value=usernames_head)
    
    df.insert(loc=df.shape[1], column='usernames_tail', value=usernames_tail)
    df.insert(loc=df.shape[1], column='followers_head', value=followers_head)
    df.insert(loc=df.shape[1], column='followers_tail', value=followers_tail)
    
    df.insert(loc=df.shape[1], column='users_keywords', value=users_keywords)
    
    # drop tweets with missing metadata. remaining data length 3451
    df = df.dropna()
    df = tokenize_and_encode_pandas_pair(df, max_length=128)
    
    datalength0 = df.shape[0]
    train_index = round (TRAINING_RATIO * datalength0)
    
    ''' ========== Shuffle and split SRQ dataframe rows ========== '''
    df = df.sample(frac=1)
    train_set = df.iloc[0:train_index].copy() # length 2752
    test_set  = df.iloc[train_index:].copy() # length 688
    
    torch.save(train_set, './data/train_set_128_individual_bertweet_bert_keywords_'+str(NUMBER)+'.bin')
    torch.save(test_set, './data/test_set_128_individual_bertweet_bert_keywords_'+str(NUMBER)+'.bin')

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
