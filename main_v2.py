#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:50:00 2021
@author: jakeyap
"""

import dataloader_utilities as dataloader
import tokenizer_v2

from classifier_models import my_ModelA0, my_ModelB0, SelfAdjDiceLoss
from transformers import BertConfig

# default imports
import torch
import torch.optim as optim
import numpy as np
import sklearn
import logging, sys, argparse
import time
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support as f1_help
from sklearn.metrics import confusion_matrix

'''
{   
     index in original data,
     x (encoded tweet), 
     token_typed_ids,
     attention_masks,
     times_labeled
     y (true label),
     interaction
}
'''

def main():
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    torch.cuda.empty_cache()
    
    time1 = time.time()
    args = get_args()
    TRNG_MB_SIZE =  args.batch_train
    TEST_MB_SIZE =  args.batch_test
    EPOCHS =        args.epochs
    LEARNING_RATE = args.learning_rate
    OPTIM =         args.optimizer
    
    DO_TRAIN =      args.do_train
    DO_TEST =       args.do_test
    
    MODEL_NAME =    args.model_name
    EXP_NAME =      args.exp_name
    
    #DEV_DATA =      args.dev_data
    TRAIN_DATA =    args.train_data
    TEST_DATA =     args.test_data
    KFOLDS =        args.k_folds
    FOLDS2RUN =     args.folds2run
    
    DEBUG =         args.debug
    LOG_INTERVAL =  args.log_interval
    ''' ===================================================='''
    ''' ---------- Parse addtional arguments here ----------'''
    LOSS_FN =       args.loss_fn
    W_SAMPLE =      args.w_sample
    ''' ===================================================='''
    
    model_savefile = './log_files/saved_models/'+EXP_NAME+'_'+MODEL_NAME+'.bin'   # to save/load model from
    plotfile = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.png'            # to plot losses
    if DO_TRAIN:
        logfile_name = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.log'    # for recording training progress
    else:
        logfile_name = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.test'
    
    
    file_handler = logging.FileHandler(filename=logfile_name)       # for saving into a log file
    stdout_handler = logging.StreamHandler(sys.stdout)              # for printing onto terminal
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt= '%m/%d/%Y %H:%M:%S', handlers=handlers, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('----------------- Hyperparameters ------------------')
    logger.info('======== '+MODEL_NAME+' =========')
    logger.info('===== Hyperparameters ======')
    logger.info('batch_train: %d' % TRNG_MB_SIZE)
    logger.info('batch_test:  %d' % TEST_MB_SIZE)
    logger.info('epochs: %d ' % EPOCHS)
    logger.info('learning_rate: %1.6f' % LEARNING_RATE)
    logger.info('optimizer: ' + OPTIM)
    logger.info('debug: ' + str(DEBUG))
    
    logger.info('------------------ Getting model -------------------')
    model = get_model(logger,MODEL_NAME)
    model.cuda()
    #model.resize_token_embeddings(len(tokenizer_utilities.tokenizer))
    
    logger.info('--------------- Getting dataframes -----------------')
    test_df = torch.load(TEST_DATA)
    full_train_df = torch.load(TRAIN_DATA)
    
    if DEBUG:
        test_df = test_df[0:20]
        full_train_df = test_df
        
    
    if DO_TRAIN:
        logger.info('------- Setting loss function and optimizer --------')
        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        if LOSS_FN == 'dice':
            loss_fn = SelfAdjDiceLoss(reduction='mean')
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_df_len = len(full_train_df)   # figure out training data length
        fold_size = train_df_len // KFOLDS  # figure out fold size
        
        ''' Manual split for now '''
        train_df = full_train_df[fold_size:]
        dev_df = full_train_df[:fold_size]
        #logger.info('------------ Split into kfolds %d / %d--------------' % (fold+1,KFOLDS))
        #dev_df = full_train_df[fold*fold_size : (fold+1)*fold_size]
        # train_df = full_train_df[0: ]
        #dev_df = torch.load(DEV_DATA)
        # TODO: implement a function to split the data separately
        # TODO: implement kfolding?
    
        logger.info('------------ Converting to dataloaders -------------')
        
        train_dl = dataloader.df_2_dl_v2(train_df, TRNG_MB_SIZE, randomize=True, weighted_sample=W_SAMPLE, logger=logger)
        test_dl = dataloader.df_2_dl_v2(test_df, TEST_MB_SIZE, randomize=False)
        dev_dl = dataloader.df_2_dl_v2(dev_df, TEST_MB_SIZE, randomize=False)
    
        logger.info('---------------- Starting training -----------------')
        train(model=model, train_dl=train_dl, dev_dl=dev_dl, 
              logger=logger, log_interval=LOG_INTERVAL, epochs=EPOCHS,
              loss_fn=loss_fn, optimizer=optimizer, 
              plotfile=plotfile, modelfile=model_savefile)
        
    # regardless of do_train or not, reload best models
    saved_params = torch.load(model_savefile)
    model.load_state_dict(saved_params)
    results = test(model=model, 
                   dataloader=test_dl,
                   logger=logger,
                   log_interval=LOG_INTERVAL,
                   print_string='test')
    y_pred = results[0]
    y_true = results[1]
    #logits = results[2]
    
    
    f1_metrics = f1_help(y_true, y_pred,    # calculate f1 scores
                             average=None,      # dont set to calculate for all
                             labels=[0,1,2,3])  # number of classes
    precisions, recalls, f1scores, supports = f1_metrics
    accuracy = calculate_acc(y_pred, y_true)
    msg = f1_metrics_msg(precisions, recalls, f1scores, supports, accuracy)
    
    logger.info(msg)
    time2 = time.time()
    print_time(time1, time2, logger)
    
    return

def get_model(logger=None, modelname=''):
    '''
    Finds a model and returns it. 

    Returns
    -------
    model object.
    
    '''
    if modelname=='my_modelA0':
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 4
        model = my_ModelA0(config)
    else:
        msg = 'model not found, exiting ' + modelname
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        raise Exception
    model.resize_token_embeddings(len(tokenizer_v2.tokenizer))
    return model

def train(model, train_dl, dev_dl, logger, log_interval, epochs, loss_fn, optimizer, plotfile, modelfile):
    losses = []
    loss_horz = []
    dev_losses=[]
    dev_loss_horz=[]
    f1_scores = []
    f1_horz = []
    best_f1 = -1
    
    gpu = torch.device("cuda")
    
    for epoch in range(epochs):
        model.train()   # set model into training mode
        for batch_id, minibatch in enumerate(train_dl):
            if batch_id % log_interval == 0:
                logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch, batch_id)))

            #x0 = minibatch[0].to(gpu)   # index in orig data
            x1 = minibatch[1].to(gpu)   # encoded tokens
            x2 = minibatch[2].to(gpu)   # token_type_ids 
            x3 = minibatch[3].to(gpu)   # attention_mask 
            #x4 = minibatch[4].to(gpu)   # times_labeled
            
            #y = minibatch[5].to(gpu)    # true label 6 class
            y = minibatch[6].to(gpu)    # true label 4 class
            
            outputs = model(input_ids=x1,    # shape=(n,C) where n=batch size
                            attention_mask=x3, 
                            token_type_ids=x2)
            
            logits = outputs[0]
            loss = loss_fn(logits, y)   # calculate the loss
            loss.backward()             # backward prop
            optimizer.step()            # step the gradients once
            optimizer.zero_grad()       # clear gradients before next step
            loss_value = loss.item()    # get value of loss
            losses.append(loss_value)   # archive the loss
            
            if len(loss_horz)==0:
                loss_horz.append(0)
            else:
                loss_horz.append(len(loss_horz))
        model.eval()    # change back to eval mode
        results = test(model=model, 
                       dataloader=dev_dl,
                       logger=logger,
                       log_interval=log_interval,
                       print_string='dev')
        
        y_pred = results[0]
        y_true = results[1]
        logits = results[2]
        dev_loss = loss_fn(logits, y_true)
        dev_loss_value = dev_loss.item()
        dev_losses.append(dev_loss_value)
        dev_loss_horz.append(loss_horz[-1])
        
        f1_metrics = f1_help(y_true, y_pred,    # calculate f1 scores
                             average=None,      # dont set to calculate for all
                             labels=[0,1,2,3])  # number of classes
        precisions, recalls, f1scores, supports = f1_metrics
        accuracy = calculate_acc(y_pred, y_true)
        msg = f1_metrics_msg(precisions, recalls, f1scores, supports, accuracy)
        logger.info(msg)
        
        f1_score = sum(f1scores) / len(f1scores)
        f1_scores.append(f1_score)
        f1_horz.append(epoch)
        if f1_score > best_f1:      # if best f1 score is reached
            best_f1 = f1_score      # store best score
            torch.save(model.state_dict(), 
                       modelfile)   # save model
        
    state = torch.load(modelfile)   # reload best model
    model.load_state_dict(state)
    fig, axes = plt.subplots(2,1)
    ax0 = axes[0] 
    ax0.scatter(loss_horz, losses)
    ax0.scatter(dev_loss_horz, dev_losses)
    ax0.set_ylabel('Training losses')
    ax0.set_ylabel('Training loss')
    ax0.grid(True)
    ax1 = axes[1]
    ax1.scatter(f1_horz, f1_scores)
    ax1.set_ylabel('F1 score')
    ax1.grid(True)
    fig.savefig(plotfile)
    return

def test(model, dataloader, logger, log_interval, print_string='test'):
    '''
    Runs a test on data insider dataloader

    Parameters
    ----------
    model : pytorch neural network module
        Generic NN.
    dataloader : pytorch dataloader
        data to run tests on.
    logger : python logger
        For writing info.
    log_interval : int
        For deciding how often to print onto log file.

    Returns
    -------
    y_pred : linear tensor
        linear array of predicted values
    y_true : linear tensor
        linear array of actual labels
    y_logits : tensor 
        tensor of original output logits
    '''
    
    model.eval()
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    y_true = None
    y_pred = None
    all_logits = None
    
    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
            if batch_id % log_interval == 0:
                logger.info(('\tTesting '+print_string+' Minibatch: %4d' % batch_id))
            
            #x0 = minibatch[0].to(gpu)   # index in orig data
            x1 = minibatch[1].to(gpu)   # encoded tokens
            x2 = minibatch[2].to(gpu)   # token_type_ids 
            x3 = minibatch[3].to(gpu)   # attention_mask 
            #x4 = minibatch[4].to(gpu)   # times_labeled
            #y = minibatch[5].to(gpu)    # true label 6 class
            y = minibatch[6].to(gpu)    # true label 4 class
            
            outputs = model(input_ids=x1,    # shape=(n,C) where n=batch size
                            attention_mask=x3, 
                            token_type_ids=x2)   
            logits = outputs[0]
            if y_true is None:                      # for handling 1st minibatch
                y_true = y.clone().to(cpu)          # shape=(n)
                index = logits.argmax(1)            # for finding index of max value
                y_pred = index.clone().to(cpu)
                all_logits = logits.clone().to(cpu)
            else:                                   # for all other minibatches
                y_true = torch.cat((y_true,         # shape=(n,)
                                    y.clone().to(cpu)),
                                   0)
                index = logits.argmax(1)            # for finding index of max value
                y_pred = torch.cat((y_pred,         # shape=(n,)
                                    index.clone().to(cpu)),
                                   0)
                all_logits = torch.cat((all_logits, 
                                        logits.clone().to(cpu)),
                                       0)
    return [y_pred, y_true, all_logits] # both have shape of (n,)

def test_single_example(model, dataloader, logger, log_interval, index=-1, show=True):
    model.eval()
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    if index==-1:   # generate a random number
        datalen = len(dataloader)
        index = np.random.randint(0, datalen)
    
    for batch_id, minibatch in enumerate(dataloader):
        batch_to_check = index // len(minibatch)
        batch_index = index % len(minibatch)
        break
    
    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
            if batch_to_check == batch_id:
                print('Found batch number %d. Running test' % batch_id)
                
                x0 = minibatch[1].to(gpu)   # encoded tokens
                x1 = minibatch[2].to(gpu)   # token_type_ids 
                x2 = minibatch[3].to(gpu)   # attention_mask 
                
                y = minibatch[5].to(gpu)
                logits = model(input_ids=x0,    # shape=(n,C) where n=batch size
                               attention_mask=x2, 
                               token_type_ids=x1)   
                logits = logits.to(cpu)
                y_pred = logits.argmax(1)
                orig_label = y[batch_index]
                prediction = y_pred[batch_index]
                if show:
                    print('Original tweet index:')
                    print(index)
                    print('Original sentences:')
                    print(tokenizer_v2.tokenizer.decode(x0.tolist()))
                    print('Original Label: \t', orig_label)
                    print('Predicted Label: \t', prediction)
                return [y_pred, y, index]
    
def f1_metrics_msg(precisions, recalls, f1scores, supports, accuracy):
    macro_f1_score = sum(f1scores) / len(f1scores)
    weighted_f1_score = np.sum(f1scores * supports) / supports.sum()
    string = '\nLabels \tPrec. \tRecall\tF1    \tSupp  \n'
    string +='Denial \t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[0], recalls[0], f1scores[0], supports[0])
    string +='Support\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[1], recalls[1], f1scores[1], supports[1])
    string +='Comment\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[2], recalls[2], f1scores[2], supports[2])
    string +='Queries\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[3], recalls[3], f1scores[3], supports[3])
    string +='MacroF1\t%1.4f\n' % macro_f1_score
    string +='F1w_avg\t%1.4f\n' % weighted_f1_score
    string +='Acc    \t%2.1f\n' % (accuracy * 100)
    return string

def calculate_acc(y_pred, y_true):
    correct = y_pred == y_true
    length = len(y_pred)
    return correct.sum().item() / length

def print_time(old_time, new_time, logger=None):
    '''
    Prints time in hh mm ss format

    Parameters
    ----------
    old_time : time in seconds
        DESCRIPTION.
    new_time : time in seconds
        DESCRIPTION.
    logger : logger object, optional
        Logger object to print to. If None, print to default console

    Returns
    -------
    None.
    '''
    hours = (new_time-old_time) // 3600
    remain = (new_time-old_time) % 3600
    minutes = remain // 60
    seconds = remain % 60
    string = 'Time taken: %dh %dm %2ds' % (hours, minutes, seconds)
    if logger is None:
        print(string)
    else:
        logger.info(string)

def get_args():
    '''
    Gets the arguments from command line

    Returns
    -------
    args dictionary

    '''
    parser = argparse.ArgumentParser()
    # Generic stuff first
    parser.add_argument("--batch_train",    default=1, type=int, help="minibatch size for training")
    parser.add_argument("--batch_test",     default=1, type=int, help="minibatch size for testing")
    parser.add_argument("--epochs",         default=1, type=int, help="num of training epochs")
    parser.add_argument("--learning_rate",  default=1, type=float,help="learning rate")
    parser.add_argument("--optimizer",      default="adam",      help="adam or rmsprop")
    
    parser.add_argument("--do_train",       action="store_true", help="Whether to run training")
    parser.add_argument("--do_test",        action="store_true", help="Whether to run tests")
    
    parser.add_argument("--model_name",     default="my_modelA0",help="model name")
    parser.add_argument("--exp_name",       default="expXX",     help="Log filename prefix")
    
    parser.add_argument("--train_data",     default='./data/train_set_256_new.bin')
    parser.add_argument("--test_data",      default='./data/test_set_256_new.bin')
    
    parser.add_argument("--k_folds",        default=4, type=int, help='number of segments to fold training data')
    parser.add_argument("--folds2run",      default=1, type=int, help='number of times to do validation folding')
    
    parser.add_argument("--debug",          action="store_true", help="Debug flag")
    parser.add_argument("--log_interval",   default=1, type=int, help="num of batches before printing")
    ''' ===================================================='''
    ''' ========== Add additional arguments here ==========='''
    parser.add_argument('--loss_fn',        default='ce_loss',  help='loss function. ce_loss (default) or dice_loss')
    parser.add_argument('--w_sample',       action='store_true',help='non flat sampling of training examples')
    ''' ===================================================='''
    return parser.parse_args()


if __name__ == '__main__':
    main()