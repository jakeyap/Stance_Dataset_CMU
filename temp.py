#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:31:47 2020

@author: jakeyap
"""


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
