#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:17:12 2021

@author: jakeyap
"""

'''
from transformers import BertTokenizer, BertForMultipleChoice
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)

tmp = {k: v.unsqueeze(0) for k,v in encoding.items()}
outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
#loss = outputs.loss
#logits = outputs.logits
'''

''' to check whether the weighted sampler is doing its job correctly across batch sizes edge cases '''
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
import matplotlib.pyplot as plt

t1 = torch.ones((1000), dtype=torch.long)
t1.shape
t2 = torch.ones((100), dtype=torch.long) * 2
t2[0]
t3 = torch.ones((10), dtype=torch.long) * 3
t4 = torch.cat((t1, t2, t3), dim=0)

dataset = torch.utils.data.TensorDataset(t4)
topic_counts = [1000, 100, 10]
topic_weights = 1.0 / torch.tensor(topic_counts, dtype=torch.float) # inverse to get class weight
sample_weights = topic_weights[t4.long()-1]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
#sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=False)

list1 = []
list2 = []
list3 = []

dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=1000, sampler=sampler)
dataloader2 = torch.utils.data.DataLoader(dataset, batch_size=100, sampler=sampler)
dataloader3 = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=sampler)

counter = 0
for i, batch in enumerate(dataloader1):
    tensor = batch[0]
    list1.extend(tensor.tolist())
    counter += 1
    
print(counter)
counter = 0
for i, batch in enumerate(dataloader2):
    tensor = batch[0]
    list2.extend(tensor.tolist())
    counter += 1

print(counter)
counter = 0
for i, batch in enumerate(dataloader3):
    tensor = batch[0]
    list3.extend(tensor.tolist())
    counter += 1
print(counter)

plt.figure()
#plt.hist([list1, list2, list3], rwidth=1/3, color=['red','blue','green'])
plt.hist([list1, list2, list3], bins=[0.5,1.5,2.5,3.5], color=['red','blue','green'])
