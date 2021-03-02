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
    
    new_df = dataframe
    posts_index     = new_df.index.values
    posts_index     = posts_index.reshape(((-1,1)))
    #print(new_df['encoded_tweets'])
    
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
                                            num_samples=len(sample_weights),
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


def df_2_dl_pretrain(dataframe, 
                     batch_size=64,
                     randomize=False,
                     weighted_sample=False,
                     DEBUG=False,
                     logger=None):
    """
    FOR PRETRAINING ONLY!!!
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
        event
        target_text
        response_text
        encoded_tweets
        token_type_ids
        attention_mask
        swapped
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
    
    encoded_tweets  = new_df['encoded_tweets'].values.tolist()
    encoded_tweets  = torch.stack(encoded_tweets, dim=0).squeeze(1)
    token_type_ids  = new_df['token_type_ids'].values.tolist()
    token_type_ids  = torch.stack(token_type_ids, dim=0).squeeze(1)
    attention_mask  = new_df['attention_mask'].values.tolist()
    attention_mask  = torch.stack(attention_mask, dim=0).squeeze(1)
    swapped         = new_df['swapped'].values
    swapped         = swapped.reshape((-1))
    
    event           = new_df['event'].values
    event           = event.reshape((-1))
    #orig_length     = dataframe['orig_length'].values.reshape((-1,1))
    
    # convert numpy arrays into torch tensors
    posts_index     = torch.from_numpy(posts_index)
    #encoded_tweets  = torch.from_numpy(encoded_tweets)
    #token_type_ids  = torch.from_numpy(token_type_ids)
    #attention_mask  = torch.from_numpy(attention_mask)
    swapped         = torch.from_numpy(swapped)
    
    dataset = TensorDataset(posts_index,
                            encoded_tweets,
                            token_type_ids,
                            attention_mask,
                            swapped)
    if randomize:
        # Do shuffle here
        if weighted_sample:
            topic_counts = [0,0,0,0]
            topic_labels = ['General_Terms','Iran_Deal','Santa_Fe_Shooting','Student_Marches']
            for i in range(len(topic_counts)):              # count the num of each class in dataset
                count = sum(event==topic_labels[i]) + 1     # min counts of 1 for stability during debug
                topic_counts[i] = count
                if logger is None:
                    print(topic_labels[i] + ': ' + str(topic_counts[i]))
                else:
                    logger.info(topic_labels[i] + ': ' + str(topic_counts[i]))
            topic_weights = 1.0 / torch.tensor(topic_counts, dtype=torch.float) # inverse to get class weight
            topic_index = (event==topic_labels[0]) * 0
            for i in [1,2,3]:
                topic_index += (event==topic_labels[i]) * i
            sample_weights = topic_weights[topic_index] # for each sample, adjust its sample weight
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
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


def df_2_dl_v3(dataframe, 
               batch_size=64,
               randomize=False,
               weighted_sample=False,
               weight_attr='stance',
               viral_attr='likes',
               viral_threshold=80,
               DEBUG=False,
               logger=None):
    """
    Converts a dataframe into a DataLoader object, and return DataLoader
    This version includes meta data to count number of likes and retweets. 
    For the TRAINING SET, if you want to split virality based on LIKES, the thresholds are as follows
    10% : 4
    20% : 22
    30% : 54
    50% : 291.5
    70% : 1415
    80% : 3556
    90% : 11236
    
    For the TRAINING SET, if you want to split virality based on RETWEETS, the thresholds are as follows
    10% : 1
    20% : 8
    30% : 24
    50% : 128.5
    70% : 587
    80% : 1417
    90% : 4260
    
    
    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe that contains all the raw tweets & encoded information.
    batch_size : int, optional
        Minibatch size to spit out. The default is 64.
    randomize : boolean, optional
        Decides whether to shuffle samples when creating minibatches. 
        The default is False.
    weighted_sample : boolean, optional
        Decides whether to use weights for sampling. Default is False
    weight_attr : string, default is 'stance'. Must be 1 of ['stance','likes','retweets']
        Used when weighted_sample==True. Choose how to weigh the sampling
    viral_attr : string, default is 'likes'. 
        Decides what virality metric to use. Must be 1 of ['likes','retweets']
    viral_threshold : int. Default is 80
        Percentile thresholds to categorize like/retweet data. Above is viral, below is not viral
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
        retweets_count
        favorite_count
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
        viral_score (retweet_count or favorite_count binary class)
    }
    """
    
    new_df = dataframe
    posts_index     = new_df.index.values
    posts_index     = posts_index.reshape(((-1,1)))
    #print(new_df['encoded_tweets'])
    
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
    
    retweets_count  = new_df['retweets_count'].values
    retweets_count  = retweets_count.reshape((-1))
    favorite_count  = new_df['favorite_count'].values
    favorite_count  = favorite_count.reshape((-1))
    
    # convert numpy arrays into torch tensors
    posts_index     = torch.from_numpy(posts_index)
    #encoded_tweets  = torch.from_numpy(encoded_tweets)
    #token_type_ids  = torch.from_numpy(token_type_ids)
    #attention_mask  = torch.from_numpy(attention_mask)
    number_labels_6 = torch.from_numpy(number_labels_6)
    number_labels_4 = torch.from_numpy(number_labels_4)
    times_labeled   = torch.from_numpy(times_labeled)
    retweets_count  = torch.from_numpy(retweets_count)
    favorite_count  = torch.from_numpy(favorite_count)
    
    
    if viral_attr=='likes':
        threshold = np.percentile(favorite_count, viral_threshold)
        viral_score = favorite_count > threshold
    elif viral_attr=='retweets':
        threshold = np.percentile(retweets_count, viral_threshold)
        viral_score = retweets_count > threshold
    
    dataset = TensorDataset(posts_index,
                            encoded_tweets,
                            token_type_ids,
                            attention_mask,
                            times_labeled,
                            number_labels_6,
                            number_labels_4,
                            viral_score)
    if randomize:
        # Do shuffle here
        if weighted_sample:
            if weight_attr=='stance':                           # For handling STANCE
                class_counts = [0,0,0,0]
                for i in range(len(class_counts)):              # for each STANCE class 
                    count = (number_labels_4==i).sum().item()   # count the num of examples
                    class_counts[i] = count                     # store the count
                class_weights = 1.0 /torch.tensor(class_counts, # inverse to get class weight
                                                  dtype=torch.float) 
                sample_weights = class_weights[number_labels_4] # for each sample, adjust its sample weight
                    
            elif weight_attr in ['likes', 'retweets'] :         # For handling LIKES and RETWEETS
                class_counts = [0,0]
                for i in range(len(class_counts)):              # for each VIRAL class 
                    count = (viral_score==i).sum().item()       # count the num of examples
                    class_counts[i] = count                     # store the count
                class_weights = 1.0 /torch.tensor(class_counts, # inverse to get class weight
                                                  dtype=torch.float) 
                sample_weights = class_weights[viral_score]     # for each sample, adjust its sample weight
                
            else:                                               # When type of label to weigh by is invalid
                raise Exception('Sampling weight type not found: ' + weight_attr)
            
            for i in range(len(class_counts)):              # count the num of each class in dataset
                if logger is None:
                    print('class: %4d \tcount: %4d' % (i, class_counts[i]))
                else:
                    logger.info('class: %4d \tcount: %4d' % (i, class_counts[i]))
            
            #class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float) # inverse to get class weight
            #sample_weights = class_weights[number_labels_4] # for each sample, adjust its sample weight
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
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
