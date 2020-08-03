#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:04:58 2020

@author: jakeyap
"""
import logging
import time

from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def dataframe_2_dataloader(dataframe, 
                           batch_size=64,
                           randomize=False,
                           DEBUG=False):
    """
    Converts a dataframe into a DataLoader object, and return DataLoader

    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe that contains all the raw tweets & encoded information.
    batch_size : int, optional
        Minibatch size to spit out. The default is 64.
    randomize : boolean, optional
        Decides whether to shuffle samples when creating minibatches. 
        The default is False.
    DEBUG : boolean, optional
        Flag to pring debugging messages. The default is False.

    Returns
    -------
    DataLoader object.
    
    Each dataframe has the following columns 
    {
        response_id
        target_id
        interaction_type
        label
        event
        target_text
        target_created_at
        response_text
        response_created_at
        Times_Labeled
        encoded_tweets
        token_type_ids
        attention_mask
        number_labels
    }
    
    Each dataloader is packed into the following tuple
    {   
         index in original data,
         x (encoded tweet), 
         token_typed_ids,
         attention_masks,
         times_labeled
         y (true label)
    }
    """
    if randomize:
        # Do shuffle here
        new_df = dataframe.sample(frac=1)
    else:
        new_df = dataframe
    
    posts_index     = new_df.index.values
    posts_index     = posts_index.reshape(((-1,1)))
    encoded_tweets  = new_df['encoded_tweets'].values
    encoded_tweets  = np.array(encoded_tweets.tolist())
    token_type_ids  = new_df['token_type_ids'].values
    token_type_ids  = np.array(token_type_ids.tolist())
    attention_mask  = new_df['attention_mask'].values
    attention_mask  = np.array(attention_mask.tolist())
    times_labeled   = new_df['Times_Labeled'].values
    times_labeled   = times_labeled.reshape(((-1,1)))
    number_labels   = new_df['number_labels'].values
    number_labels   = number_labels.reshape((-1))
    interaction     = new_df['interaction_type_num'].values
    interaction     = interaction.reshape((-1,1))
    
    # convert numpy arrays into torch tensors
    posts_index     = torch.from_numpy(posts_index)
    encoded_tweets  = torch.from_numpy(encoded_tweets)
    token_type_ids  = torch.from_numpy(token_type_ids)
    attention_mask  = torch.from_numpy(attention_mask)
    number_labels   = torch.from_numpy(number_labels)
    times_labeled   = torch.from_numpy(times_labeled)
    interaction     = torch.from_numpy(interaction)
    
    dataset = TensorDataset(posts_index,
                            encoded_tweets,
                            token_type_ids,
                            attention_mask,
                            times_labeled,
                            number_labels,
                            interaction)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=randomize)
    
    return dataloader
    
if __name__ =='__main__':
    time_start = time.time()
    NUM_TO_PROCESS = 1000000

    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)
