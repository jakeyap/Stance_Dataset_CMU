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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler

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

def df_2_dl_v2(dataframe, 
               batch_size=64,
               randomize=False,
               weighted_sample=False,
               DEBUG=False,
               logger=None):
    """
    Converts a dataframe into a DataLoader object, and return DataLoader
    This is a new version to account for 4-class-labels and 6-class-labels in data
    
    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe that contains all the raw tweets & encoded information.
    batch_size : int, optional
        Minibatch size to spit out. The default is 64.
    randomize : boolean, optional
        Decides whether to shuffle samples when creating minibatches. 
        The default is False.
    weighted_sample : boolean optional
        Decides whether to use weights for sampling. Default is False
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
        number_labels_6
        number_labels_4
    }
    
    Each dataloader is packed into the following tuple
    {   
         index in original data,
         x (encoded tweet), 
         token_typed_ids,
         attention_masks,
         times_labeled
         y (true label 6 class)
         y (true label 4 class)
    }
    """
    '''
    if randomize:
        # Do shuffle here
        new_df = dataframe.sample(frac=1)
    else:
        new_df = dataframe
    '''
    new_df = dataframe
    posts_index     = new_df.index.values
    posts_index     = posts_index.reshape(((-1,1)))
    #print(new_df['encoded_tweets'])
    '''
    encoded_tweets  = new_df['encoded_tweets'].values
    encoded_tweets  = np.array(encoded_tweets.tolist())
    token_type_ids  = new_df['token_type_ids'].values
    token_type_ids  = np.array(token_type_ids.tolist())
    attention_mask  = new_df['attention_mask'].values
    attention_mask  = np.array(attention_mask.tolist())
    times_labeled   = new_df['Times_Labeled'].values
    times_labeled   = times_labeled.reshape(((-1,1)))
    number_labels_6 = new_df['number_labels_6_types'].values
    number_labels_6 = number_labels_6.reshape((-1))
    number_labels_4 = new_df['number_labels_4_types'].values
    number_labels_4 = number_labels_4.reshape((-1))
    '''
    
    encoded_tweets  = new_df['encoded_tweets'].values.tolist()
    encoded_tweets  = torch.stack(encoded_tweets, dim=0).squeeze(1)
    token_type_ids  = new_df['token_type_ids'].values.tolist()
    token_type_ids  = torch.stack(token_type_ids, dim=0).squeeze(1)
    attention_mask  = new_df['attention_mask'].values.tolist()
    attention_mask  = torch.stack(attention_mask, dim=0).squeeze(1)
    times_labeled   = new_df['Times_Labeled'].values
    times_labeled   = times_labeled.reshape(((-1,1)))
    number_labels_6 = new_df['number_labels_6_types'].values
    number_labels_6 = number_labels_6.reshape((-1))
    number_labels_4 = new_df['number_labels_4_types'].values
    number_labels_4 = number_labels_4.reshape((-1))
    #orig_length     = dataframe['orig_length'].values.reshape((-1,1))
    
    # convert numpy arrays into torch tensors
    posts_index     = torch.from_numpy(posts_index)
    #encoded_tweets  = torch.from_numpy(encoded_tweets)
    #token_type_ids  = torch.from_numpy(token_type_ids)
    #attention_mask  = torch.from_numpy(attention_mask)
    number_labels_6 = torch.from_numpy(number_labels_6)
    number_labels_4 = torch.from_numpy(number_labels_4)
    times_labeled   = torch.from_numpy(times_labeled)
    
    dataset = TensorDataset(posts_index,
                            encoded_tweets,
                            token_type_ids,
                            attention_mask,
                            times_labeled,
                            number_labels_6,
                            number_labels_4)
    if randomize:
        # Do shuffle here
        if weighted_sample:
            class_counts = [0,0,0,0]
            for i in range(len(class_counts)):              # count the num of each class in dataset
                count = (number_labels_4==i).sum().item()   
                class_counts[i] = count
                if logger is None:
                    print(count)
                else:
                    logger.info(count)
            class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float) # inverse to get class weight
            sample_weights = class_weights[number_labels_4] # for each sample, adjust its sample weight
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=batch_size,
                                            replacement=True)
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4)
    return dataloader

    
if __name__ =='__main__':
    time_start = time.time()
    NUM_TO_PROCESS = 1000000

    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)
