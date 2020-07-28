#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:31:47 2020

This model tries to do pairwise classification in one shot

i.e. convert [category1:category2 ] into [category1 x 10 + category2]
@author: jakeyap
"""
import tokenizer_utilities
import dataloader_utilities as dataloader
import time
import torch
import torch.optim as optim
from classifier_models import my_ModelA0, my_ModelA1, my_ModelA2
from transformers import BertConfig
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
time_start = time.time()
    
FROM_SCRATCH = False # True if start loading model from scratch
TRAIN = False # True if you want to train the network. False to just test

'''======== FILE NAMES FOR LOGGING ========'''
iteration = 2
MODELNAME = 'modelA2'

ITER1 = str(iteration)
DATADIR = './data/'
MODELDIR= './models/'
RESULTDIR='./results/'
# For storing the tokenized posts
train_file = DATADIR + "train_set.bin"
test_file = DATADIR + "test_set.bin"

load_model_file = MODELDIR+MODELNAME+"_model_"+ITER1+".bin"
load_config_file = MODELDIR+MODELNAME+"_config_"+ITER1+".bin"
load_optstate_file = MODELDIR+MODELNAME+"_optimizer_"+ITER1+".bin"
load_losses_file = RESULTDIR+MODELNAME+"_losses_"+ITER1+".bin"

# Put a timestamp saved states so that overwrite accidents are less likely
timestamp = time.time()
timestamp = str("%10d" % timestamp)

ITER2 = str(iteration+1)
save_model_file = MODELDIR+MODELNAME+"_model_"+ITER2+"_"+timestamp+".bin"
save_config_file = MODELDIR+MODELNAME+"_config_"+ITER2+"_"+timestamp+".bin"
save_optstate_file = MODELDIR+MODELNAME+"_optimizer_"+ITER2+"_"+timestamp+".bin"
save_losses_file = RESULTDIR+MODELNAME+"_losses_"+ITER2+"_"+timestamp+".bin"

'''======== HYPERPARAMETERS START ========'''
NUM_TO_PROCESS = 100000
BATCH_SIZE_TRAIN = 40
BATCH_SIZE_TEST = 40
LOG_INTERVAL = 10

N_EPOCHS = 80
LEARNING_RATE = 0.001
MOMENTUM = 0.5

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

cpu = torch.device('cpu')
gpu = torch.device('cuda')
DEVICE = cpu

# Load saved data
print('Loading data')
train_set_df = torch.load(train_file)
tests_set_df = torch.load(test_file)

train_loader = dataloader.dataframe_2_dataloader(train_set_df, 
                                                 batch_size=BATCH_SIZE_TRAIN,
                                                 randomize=True,
                                                 DEBUG=False)

test_loader = dataloader.dataframe_2_dataloader(tests_set_df,
                                                batch_size=BATCH_SIZE_TEST,
                                                randomize=False,
                                                DEBUG=False)

# count the number in the labels
labels = train_set_df['label']
label_counts = torch.zeros(size=(1,6), dtype=float)
for label in labels:
    labelnum = tokenizer_utilities.convert_label_string2num(label)     
    label_counts[0,labelnum] += 1

# and for numerical stability
loss_sum = torch.sum(label_counts)
loss_weights = torch.true_divide(loss_sum, label_counts)
loss_weights = loss_weights.reshape(6).to('cuda')
loss_weights = torch.true_divide(loss_weights, loss_weights.mean())


if FROM_SCRATCH:
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 6
    model = my_ModelA0(config)
    # Move model into GPU
    model.to(gpu)
    # Define the optimizer. Use SGD
    '''
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    '''
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Variables to store losses
    train_losses = []
    train_count = []
    tests_losses = []
    tests_accuracy = []
    tests_count = []
    f1_scores = []
    
else:
    config = BertConfig.from_json_file(load_config_file)
    model = my_ModelA0(config)
    
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict)
    del state_dict # the state dict is huge potentially. delete after use
    # Move model into GPU
    model.to(gpu)
    # Define the optimizer. Use SGD
    '''
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    '''
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optim_state = torch.load(load_optstate_file)
    optimizer.load_state_dict(optim_state)
    # Variables to store losses
    losses = torch.load(load_losses_file)
    train_losses = losses[0]
    train_count = losses[1]
    tests_losses = losses[2]
    tests_accuracy = losses[3]
    tests_count = losses[4]
    f1_scores = losses[5]


# Define the loss function
#loss_weights = 
loss_function = torch.nn.CrossEntropyLoss(weight=loss_weights.float(),
                                          reduction='sum')
#loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
'''
{   
     index in original data,
     x (encoded tweet), 
     token_typed_ids,
     attention_masks,
     times_labeled
     y (true label)
}
'''
def train(epoch):
    # Set network into training mode to enable dropout
    model.train()
    
    for batch_idx, minibatch in enumerate(train_loader):
        #move stuff to gpu
        x = minibatch[1].to(gpu)
        y = minibatch[5].to(gpu)
        token_type_ids = minibatch[2].to(gpu)
        attention_mask = minibatch[3].to(gpu)
        
        # Reset gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward prop throught BERT
        outputs = model(input_ids = x,
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids)
        
        #outputs is a length=1 tuple. Get index 0 to access real outputs
        # Calculate loss
        loss = loss_function(outputs[0], y)
        # Backward prop find gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        
        #delete references to free up GPU space
        del x, y, token_type_ids, attention_mask
        del outputs
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * BATCH_SIZE_TRAIN, len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), 
                  loss.item() / BATCH_SIZE_TRAIN))
            
            train_losses.append(loss.item() / BATCH_SIZE_TRAIN)
            if len(train_count) == 0:
                train_count.append(BATCH_SIZE_TEST*LOG_INTERVAL)
            else:
                train_count.append(train_count[-1] + BATCH_SIZE_TEST*LOG_INTERVAL)
            
            # Store the states of model and optimizer into logfiles
            # In case training gets interrupted, you can load old states
            
            torch.save(model.state_dict(), save_model_file)
            torch.save(optimizer.state_dict(), save_optstate_file)
            torch.save([train_losses,
                        train_count,
                        tests_losses,
                        tests_accuracy,
                        tests_count,
                        f1_scores], save_losses_file)
            model.config.to_json_file(save_config_file)

def test(save=False):
    # This function evaluates the entire test set
    
    # Set network into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    # start the label arrays. 1st data point has to be deleted later
    predicted_label_arr = torch.tensor([[0]])
    groundtruth_arr = torch.tensor([0])
    #predicted_label_arr = torch.zeros(1, len(tests_loader.dataset))
    with torch.no_grad():
        for batchid, minibatch in enumerate(test_loader):
            x = minibatch[1].to('cuda')
            y = minibatch[5].to('cuda')
            token_type_ids = minibatch[2].to('cuda')
            attention_mask = minibatch[3].to('cuda')
            outputs = model(input_ids = x,
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
            #outputs is a length=1 tuple. Get index 0 to access real outputs
            outputs = outputs[0]
            test_loss += loss_function(outputs, y).item()
            
            predicted_label = outputs.data.max(1, keepdim=True)[1]
            correct += predicted_label.eq(y.data.view_as(predicted_label)).sum()
            
            predicted_label_arr = torch.cat((predicted_label_arr,
                                             predicted_label.to('cpu')),
                                            0)
            groundtruth_arr = torch.cat((groundtruth_arr,
                                         y.to('cpu')),
                                        0)
            #delete references to free up GPU space
            del x, y, token_type_ids, attention_mask
            del outputs, predicted_label
    test_loss /= len(test_loader.dataset)
        
    predicted_label_arr = predicted_label_arr.reshape(shape=(-1,))
    groundtruth_arr = groundtruth_arr.reshape(shape=(-1,))
    score = f1_score(groundtruth_arr, predicted_label_arr, average='macro')
    total_len = len(test_loader.dataset)
    accuracy = 100. * correct.to('cpu') / total_len
    accuracy = accuracy.item()
    print('Test set: Avg loss: {:3.4f}'.format(test_loss), end='\t')
    print(' Accuracy: {:6d}/{:6d} ({:2.1f}%)'.format(correct, total_len, accuracy), end='\t')
    print(' F1 score: {:1.2f}\n'.format(score))
    
    if save:
        tests_losses.append(test_loss)
        f1_scores.append(score)
        tests_accuracy.append(accuracy)
        if len(tests_count) == 0:
            tests_count.append(len(train_loader.dataset))
        else:
            tests_count.append(train_count[-1] + len(train_loader.dataset))
        torch.save([train_losses,
                    train_count,
                    tests_losses,
                    tests_accuracy,
                    tests_count,
                    f1_scores], 
                   save_losses_file)
    return predicted_label_arr[1:], groundtruth_arr[1:]


def eval_single_example(number_to_check, show=True):
    batch_to_check = number_to_check // BATCH_SIZE_TEST
    index_to_check = number_to_check % BATCH_SIZE_TEST
    
    with torch.no_grad():
        for batchid, minibatch in enumerate(test_loader):
            if batchid == batch_to_check:
                indices = minibatch[0]
                index = indices [index_to_check]
                x = minibatch[1].to('cuda')
                y = minibatch[5].to('cuda')
                token_type_ids = minibatch[2].to('cuda')
                attention_mask = minibatch[3].to('cuda')
                outputs = model(input_ids = x,
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids)
                
                outputs = outputs[0]
                # classification is the one with highest score output score
                prediction = outputs.argmax(dim=1)
                prediction = tokenizer_utilities.convert_label_num2string(prediction[index_to_check])
                encoded_sentence = x[index_to_check,0:]
                reallabels = tokenizer_utilities.convert_label_num2string(y[index_to_check])
                
                if show:
                    print('Original tweet index:')
                    print(index)
                    print('Original sentences:')
                    print(tokenizer_utilities.tokenizer.decode(encoded_sentence.tolist()))
                    print('Original Labels: \t', reallabels)
                    print('Predicted Labels: \t', prediction)
                
                del x, y, token_type_ids, attention_mask, outputs
                del encoded_sentence, 
                return reallabels, prediction

def plot_losses(offset=0):
    fig1 = plt.figure(1)
    try:
        losses = torch.load(save_losses_file)
    except Exception:
        losses = torch.load(load_losses_file)
    train_losses = losses[0]
    train_count = losses[1]
    tests_losses = losses[2]
    tests_accuracy = losses[3]
    tests_count = losses[4]
    f1_scores = losses[5]
    
    plt.scatter(train_count[offset:], 
                train_losses[offset:], label='Train')
    plt.scatter(tests_count, tests_losses, label='Test')
    plt.ylabel('Loss')
    plt.xlabel('Minibatches seen. Batchsize='+str(BATCH_SIZE_TEST))
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    fig2 = plt.figure(2)
    plt.subplot(2,1,1)
    plt.title('Accuracy')
    plt.scatter(tests_count,tests_accuracy)
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.title('Macro F1 score')
    plt.scatter(tests_count,f1_scores)
    plt.grid(True)
    plt.tight_layout()
    return losses

if __name__ =='__main__':
    torch.cuda.empty_cache()
    test(save=False)
    if (TRAIN):
        for epoch in range(1, N_EPOCHS + 1):
            train(epoch)
            labels = test(save=True)
    plot_losses(offset=0)

    time_end = time.time()
    time_taken = time_end - time_start
    print('Time elapsed: %6.2fs' % time_taken)