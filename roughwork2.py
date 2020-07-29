# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:11:50 2020

@author: Yong Keong
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
''' 
This starts a random network. Initialize randomly first.
'''

class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.fc1 = torch.nn.Linear(5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

'''
model1 = TheModelClass()

'''
x = torch.tensor([1.0,2.0,3.0,4.0,5.0])
x = x.reshape([-1,5])

y = model1(x)
print(y)

'''
Save the data of this network. SUpposed to give 0.5485
'''
'''
torch.save(model1.state_dict(), 'testsave.bin')

'''
model2 = TheModelClass()
state_dict = torch.load('testsave.bin')
model2.load_state_dict(state_dict)
del state_dict
y = model2(x)
print(y)
