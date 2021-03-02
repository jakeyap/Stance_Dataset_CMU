#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:31:47 2020

@author: jakeyap
"""

import torch
import numpy as np
import json
import pandas as pd

filename = './data/stance_dataset.json'
raw_list = []
with open(filename) as jsonfile:
    start = 0
    num_to_count = 1e9
    
    lines = jsonfile.readlines()       
    counter = 0                 # Variable to count the loop
    end = start + num_to_count  # End index
    errors = 0                  # Counter to store num of errors
    for line in lines:
        if (counter >= start) and (counter < end):
            thread_json = json.loads(line)
            raw_list.append(thread_json)
        if (counter >= end):
            break
        if counter % 1000 == 0:
            print('Flattening thread to pair: %00000d' % counter)
        counter = counter + 1

pd_dataframe = pd.DataFrame(raw_list)


labels = pd_dataframe['label'].values
labels = labels.tolist()
labels.sort()

labels_set = set(labels)
counts = {}
for each_label in labels_set:
    counts[each_label] = 0
    
''' Count number of labels '''
datalength = pd_dataframe.shape[0]

for row in range(datalength):
    string_label = pd_dataframe.iloc[row]['label']
    counts[string_label] = counts[string_label] + 1
    
import matplotlib.pyplot as plt
plt.bar(x=counts.keys(), height=counts.values(), width=0.25)
plt.ylabel('Counts')
plt.xlabel('Labels')
plt.title('CMU twitter dataset labels')
plt.grid(True)
plt.tight_layout()

''' ========== count the labels for both sets ========== '''
import matplotlib.pyplot as plt
import tokenizer_utilities
count3 = tokenizer_utilities.empty_label_dictionary()
count4 = tokenizer_utilities.empty_label_dictionary()
count5 = tokenizer_utilities.empty_label_dictionary()
train_set = torch.load('./data/train_set.bin')
test_set  = torch.load('./data/test_set.bin')

# Count number of labels
datalength3 = train_set.shape[0]
datalength4 = test_set.shape[0]
# Go through training data to count labels
for row in range(datalength3):
    string_label = train_set.iloc[row]['label']
    count3[string_label] = count3[string_label] + 1

# Go through test data to count labels
for row in range(datalength4):
    string_label = test_set.iloc[row]['label']
    count4[string_label] = count4[string_label] + 1 
    count5[string_label] = count5[string_label] + 1 

# Normalize both histograms
train_label_max = sum(count3.values())
test_label_max = sum(count4.values())
for each_key in count3.keys():
    count3[each_key] = count3[each_key] * 100 / train_label_max
    count4[each_key] = count4[each_key] * 100 / test_label_max

plt.figure(3)
xpts = np.arange(len(count3))
width = 0.25
plot1 = plt.subplot(2,1,1)
label_list = count3.keys()
plt.bar(x=xpts-width/2,height=count3.values(), width=width, label='train-3844')
plt.bar(x=xpts+width/2,height=count4.values(), width=width, label='test-427')
plt.xticks(xpts, label_list)
plt.ylabel('Percent')
plt.xlabel('Labels')
plt.title('CMU twitter dataset labels %')
plt.grid(True)
plt.tight_layout()
plt.legend()

plot2 = plt.subplot(2,1,2)
plt.bar(x=xpts,height=count5.values(), width=width, label='train-3844')
plt.xticks(xpts, label_list)
plt.ylabel('Count')
plt.xlabel('Labels')
plt.title('Test set counts')
plt.grid(True)
plt.tight_layout()

''' ========== save both datasets into binaries ========== '''


