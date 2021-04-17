#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:18:44 2020

@author: Yong Keong
"""

NUMBER=10
import torch
train_set = torch.load('./data/train_set_128_individual_bertweet_keywords_'+str(NUMBER)+'.bin')
test_set = torch.load('./data/test_set_128_individual_bertweet_keywords_'+str(NUMBER)+'.bin')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for df in [train_set, test_set]:
    encoded_tweets_h = []
    encoded_tweets_t = []
    
    token_type_ids_h = []
    token_type_ids_t = []
    
    attention_mask_h = []
    attention_mask_t = []
    
    encoded_keywords = []
    datalen = len(df)
    
    for i in range(datalen):
        topic = df.iloc[i]['event']                       # get event
        topic = topic.replace('_', ' ')                         # replace underscore with space
        user_h = df.iloc[i]['usernames_head']
        user_t = df.iloc[i]['usernames_tail']
        sep_token = ' [SEP] ' 
        head_txt = df.iloc[i]['target_text']      # get head's tweet text
        tail_txt = df.iloc[i]['response_text']    # get tail's tweet text
        
        keywords = df.iloc[i]['users_keywords']  # get the user keywords representation
        
        text_h = user_h + sep_token + topic + sep_token + head_txt
        text_t = user_t + sep_token + topic + sep_token + tail_txt
        encoded_dict_h = tokenizer.__call__(text=text_h,
                                            padding='max_length',
                                            truncation=True,
                                            is_split_into_words=False,
                                            max_length=128,
                                            return_tensors='pt')
        encoded_dict_t = tokenizer.__call__(text=text_h,
                                            padding='max_length',
                                            truncation=True,
                                            is_split_into_words=False,
                                            max_length=128,
                                            return_tensors='pt')
        
        keywords_encoding = tokenizer.encode(keywords,              # shape=(1,128))
                                             padding='max_length',
                                             truncation=True,
                                             is_split_into_words=False,
                                             max_length=128,
                                             return_tensors='pt')

        encoded_tweets_h.append(encoded_dict_h['input_ids'])
        token_type_ids_h.append(encoded_dict_h['token_type_ids'])
        attention_mask_h.append(encoded_dict_h['attention_mask'])
        encoded_tweets_t.append(encoded_dict_t['input_ids'])
        token_type_ids_t.append(encoded_dict_t['token_type_ids'])
        attention_mask_t.append(encoded_dict_t['attention_mask'])
        encoded_keywords.append(keywords_encoding)
        
    df['encoded_tweets_h'] = encoded_tweets_h
    df['token_type_ids_h'] = token_type_ids_h
    df['attention_mask_h'] = attention_mask_h
    df['encoded_tweets_t'] = encoded_tweets_t
    df['token_type_ids_t'] = token_type_ids_t
    df['attention_mask_t'] = attention_mask_t
    df['encoded_keywords'] = encoded_keywords
    
torch.save(train_set, './data/train_set_128_individual_bert_keywords_'+str(NUMBER)+'.bin')
torch.save(test_set, './data/test_set_128_individual_bert_keywords_'+str(NUMBER)+'.bin')