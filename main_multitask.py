#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:27:27 2021
For handling stance and virality multi task training
    
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

@author: jakeyap
"""

import dataloader_utilities as dataloader
import tokenizer_v2

from classifier_models import my_Bertweet, mtt_Bertweet, SelfAdjDiceLoss
# from transformers import BertConfig

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


# TODO8: debug code segments
# TODO9: run single training example for sanity checking

# DONE1: edit the dataloader df2dl_v3
# DONE2: edit the model to split its outputs
# DONE3: edit the train function for mtt
# DONE4: edit the test function mtt
# DONE5: edit the multi task printing
# DONE6: figure out like/retweet count
# DONE7: add a threshold for viral posts


torch.manual_seed(0)
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
    PRETRAIN =      args.pretrain_model
    EPOCHS2GIVEUP = args.epochs2giveup
    DROPOUT =       args.dropout
    V_THRESHOLD =   args.viral_threshold
    W_ATTR =        args.weight_attr 
    TASK =          args.task
    MTT_WEIGHT =    args.mtt_weight
    ''' ===================================================='''
    
    model_savefile = './log_files/saved_models/'+EXP_NAME+'_'+MODEL_NAME+'.bin'   # to save/load model from
    plotfile = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.png'            # to plot losses
    if DO_TRAIN:
        logfile_name = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.log'    # for recording training progress
    else:
        logfile_name = './log_files/'+EXP_NAME+'_'+MODEL_NAME+'.test'
    
    file_handler = logging.FileHandler(filename=logfile_name)       # for saving into a log file
    stdout_handler = logging.StreamHandler(sys.stdout)              # for printing onto terminal
    stderr_handler = logging.StreamHandler(sys.stderr)              # for printing errors onto terminal
    #sys.stderr = stdout_handler 
    handlers1 = [file_handler, stdout_handler]
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt= '%m/%d/%Y %H:%M:%S', handlers=handlers1, level=logging.INFO)
    handlers2 = [file_handler, stderr_handler]
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt= '%m/%d/%Y %H:%M:%S', handlers=handlers2, level=logging.ERROR)
    
    logger = logging.getLogger(__name__)
    logger.info('----------------- Hyperparameters ------------------')
    logger.info('======== '+MODEL_NAME+' =========')
    logger.info('===== Hyperparameters ======')
    for eachline in vars(args).items():
        logger.info(eachline)
    
    logger.info('------------------ Getting model -------------------')
    model = get_model(logger,MODEL_NAME, DROPOUT)
    model.cuda()
    model = torch.nn.DataParallel(model)
    if PRETRAIN != '':  # reload pretrained model 
        logger.info('loading pretrained model file ' + PRETRAIN)
        saved_params = torch.load(PRETRAIN)
        model.load_state_dict(saved_params)
        
    logger.info('--------------- Getting dataframes -----------------')
    test_df = torch.load(TEST_DATA)
    full_train_df = torch.load(TRAIN_DATA)
    
    if DEBUG:
        test_df = test_df[0:40]
        full_train_df = test_df
    
    if DO_TRAIN:
        logger.info('------- Setting loss function and optimizer --------')
        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        if LOSS_FN == 'dice':
            loss_fn = SelfAdjDiceLoss(reduction='mean')
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        if OPTIM=='adam':
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            raise Exception('Optimizer not found: ' + optimizer)
        
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
        #train_dl = dataloader.df_2_dl_v2(train_df, TRNG_MB_SIZE, randomize=True, weighted_sample=W_SAMPLE, logger=logger)
        #dev_dl = dataloader.df_2_dl_v2(dev_df, TEST_MB_SIZE, randomize=False)
        train_dl = dataloader.df_2_dl_v3(train_df, 
                                         batch_size=TRNG_MB_SIZE, 
                                         randomize=True, 
                                         weighted_sample=W_SAMPLE, 
                                         weight_attr=W_ATTR,
                                         viral_threshold=V_THRESHOLD, 
                                         logger=logger)
        dev_dl = dataloader.df_2_dl_v3(dev_df, 
                                       batch_size=TEST_MB_SIZE, 
                                       randomize=False, 
                                       weighted_sample=False, 
                                       viral_threshold=V_THRESHOLD, 
                                       logger=logger)
    
        logger.info('---------------- Starting training -----------------')
        train(model=model, train_dl=train_dl, dev_dl=dev_dl, 
              logger=logger, log_interval=LOG_INTERVAL, epochs=EPOCHS,
              loss_fn=loss_fn, optimizer=optimizer, 
              plotfile=plotfile, modelfile=model_savefile,
              epochs_giveup=EPOCHS2GIVEUP,
              task=TASK, 
              mtt_weight=MTT_WEIGHT)
    
        # reload best models
        saved_params = torch.load(model_savefile)
        model.load_state_dict(saved_params)
        
    test_dl = dataloader.df_2_dl_v3(test_df, TEST_MB_SIZE, randomize=False)
    results = test(model=model, 
                   dataloader=test_dl,
                   logger=logger,
                   log_interval=LOG_INTERVAL,
                   print_string='test')
    
    y_pred_s = results[0]
    y_pred_v = results[1]
    y_true_s = results[2]
    y_true_v = results[3]
    #logits_s = results[4]
    #logits_v = results[5]
    
    f1_metrics_s = f1_help(y_true_s, y_pred_s,  # calculate f1 scores for stance
                           average=None,        # dont set to calculate for all
                           labels=[0,1,2,3])    # number of classes = 4
    f1_metrics_v = f1_help(y_true_v, y_pred_v,  # calculate f1 scores for viral
                           average=None,        # dont set to calculate for all
                           labels=[0,1])        # number of classes = 2
    prec_s, rec_s, f1s_s, supp_s = f1_metrics_s
    prec_v, rec_v, f1s_v, supp_v = f1_metrics_v
    acc_s = calculate_acc(y_pred_s, y_true_s)
    acc_v = calculate_acc(y_pred_v, y_true_v)
    msg_s = f1_metrics_msg_stance(prec_s, rec_s, f1s_s, supp_s, acc_s)
    msg_v = f1_metrics_msg_viral(prec_v, rec_v, f1s_v, supp_v, acc_v)
    
    logger.info(msg_s + msg_v)
    #logger.info(msg_v)
    time2 = time.time()
    print_time(time1, time2, logger)
    
    return

def get_model(logger=None, modelname='', dropout=0.1):
    '''
    Finds a model and returns it. 

    Returns
    -------
    model object.
    
    '''
    
    if modelname=='my_Bertweet':
        #config = BertConfig.from_pretrained('bert-base-uncased')
        #config.num_labels = 4
        model = my_Bertweet(4, dropout)
    elif modelname=='mtt_Bertweet':
        model = mtt_Bertweet(4, dropout)
    else:
        msg = 'model not found, exiting ' + modelname
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        raise Exception
    
    return model

def train(model, train_dl, dev_dl, logger, log_interval, epochs, loss_fn, optimizer, plotfile, modelfile, epochs_giveup=10, task='multi', mtt_weight=1.0):
    '''
    all the params needed are straightforward. except for plotfile, modelfile, ep
    model :         pytorch neural network model
    train_dl :      training dataloader
    dev_dl :        dev set dataloader
    logger :        python logger
    log_interval :  how many epochs before printing progress
    epochs :        max number of epochs to run
    loss_fn :       loss function
    optimizer :     duh
    plotfile :      filename to save plot to
    modelfile :     filename to save model params to
    epochs_giveup : if this number of epochs pass w/o any improvements to f1 score, give up. 
    task :          what task to train on. "multi", "stance" or "viral"
    mtt_weight :    relative weight of viral : stance loss. defaults to 1
    
    Returns
    -------
    None.

    '''
    losses_v = []
    losses_s = []
    losses = []
    loss_horz = []
    
    dev_losses_v = []
    dev_losses_s = []
    dev_losses = []
    dev_loss_horz = []
    
    dev_f1_scores_v = []
    dev_f1_scores_s = []
    dev_f1_scores = []
    dev_f1_horz = []
    best_f1 = -1
    epochs_since_best = 0
    
    gpu = torch.device("cuda")
    
    for epoch in range(epochs):
        model.train()   # set model into training mode
        for batch_id, minibatch in enumerate(train_dl):
            if batch_id % log_interval == 0:
                logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch, batch_id)))

            #x0 = minibatch[0].to(gpu)  # index in orig data (unused)
            x1 = minibatch[1].to(gpu)   # encoded tokens
            x2 = minibatch[2].to(gpu)   # token_type_ids 
            x3 = minibatch[3].to(gpu)   # attention_mask 
            #x4 = minibatch[4].to(gpu)  # times_labeled (unused)
            #y = minibatch[5].to(gpu)   # true label 6 stance class (unused)
            y_s = minibatch[6].to(gpu)  # true label 4 stance class
            y_v = minibatch[7].to(gpu)  # viral_score
            
            outputs = model(input_ids=x1,    # shape=(n,C) where n=batch size
                            attention_mask=x3, 
                            token_type_ids=x2,
                            task=task)
            logits_s = outputs[0]
            logits_v = outputs[1]
            
            if task=='stance':
                loss_s = loss_fn(logits_s, y_s) # calculate the stance loss
                losses_s.append(loss_s.item())  # archive the loss
                loss = loss_s
            elif task=='viral':
                loss_v = loss_fn(logits_v, y_v) # calculate the viral loss
                losses_v.append(loss_v.item())  # archive the loss
                loss = loss_v
            elif task=='multi':
                loss_s = loss_fn(logits_s, y_s) # calculate the stance loss
                losses_s.append(loss_s.item())  # archive the loss
                loss_v = loss_fn(logits_v, y_v) # calculate the viral loss
                losses_v.append(loss_v.item())  # archive the loss
                loss = loss_s+mtt_weight*loss_v # sum the losses
            else:
                err_string = 'task not found : ' + task
                logger.info(err_string)
                raise Exception(err_string)
                
            loss.backward()             # backward prop
            optimizer.step()            # step the gradients once
            optimizer.zero_grad()       # clear gradients before next step
            loss_value = loss.item()    # get value of total loss
            losses.append(loss_value)   # archive the total loss
            
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
        
        y_pred_s = results[0]
        y_pred_v = results[1]
        y_true_s = results[2]
        y_true_v = results[3]
        logits_s = results[4]
        logits_v = results[5]
        
        dev_loss_s = loss_fn(logits_s, y_true_s)
        dev_loss_v = loss_fn(logits_v, y_true_v)
        dev_loss_value_s = dev_loss_s.item()
        dev_loss_value_v = dev_loss_v.item()
        dev_loss_value = (dev_loss_value_s + mtt_weight * dev_loss_value_v) / (1 + mtt_weight)
        
        dev_losses_s.append(dev_loss_value_s)
        dev_losses_v.append(dev_loss_value_v)
        dev_losses.append(dev_loss_value)
        dev_loss_horz.append(loss_horz[-1])
        
        f1_metrics_s = f1_help(y_true_s, y_pred_s,  # calculate f1 scores for stance
                               average=None,        # dont set to calculate for all
                               labels=[0,1,2,3])    # number of classes
        f1_metrics_v = f1_help(y_true_v, y_pred_v,  # calculate f1 scores for viral
                               average=None,        # dont set to calculate for all
                               labels=[0,1])        # number of classes
        
        prec_s, recall_s, f1s_s, supp_s = f1_metrics_s
        prec_v, recall_v, f1s_v, supp_v = f1_metrics_v
        acc_s = calculate_acc(y_pred_s, y_true_s)
        acc_v = calculate_acc(y_pred_v, y_true_v)
        msg_s = f1_metrics_msg_stance(prec_s, recall_s, f1s_s, supp_s, acc_s)
        msg_v = f1_metrics_msg_viral(prec_v, recall_v, f1s_v, supp_v, acc_v)
        logger.info(msg_s + msg_v)
        
        f1_score_s = sum(f1s_s) / len(f1s_s)
        f1_score_v = sum(f1s_v) / len(f1s_v)
        
        if task=='stance':
            f1_score = f1_score_s
        elif task=='viral':
            f1_score = f1_score_v
        else:
            f1_score = (f1_score_s + mtt_weight * f1_score_v) / (1 + mtt_weight)
        
        dev_f1_scores_s.append(f1_score_s)
        dev_f1_scores_v.append(f1_score_v)
        dev_f1_scores.append(f1_score)
        dev_f1_horz.append(epoch)
        epochs_since_best += 1
        
        if f1_score > best_f1:      # if best f1 score is reached
            logger.info('Best results so far. Saving model...')
            best_f1 = f1_score      # store best score
            epochs_since_best = 0   # reset the epochs counter
            torch.save(model.state_dict(), 
                       modelfile)   # save model
        
        if epochs_since_best >= epochs_giveup:
            logger.info('No improvements in F1 for %d epochs' % epochs_since_best)
            break                   # stop training if no improvements for too long
        
    state = torch.load(modelfile)   # reload best model
    model.load_state_dict(state)
    fig, axes = plt.subplots(2,1)
    ax0 = axes[0]
    ax1 = axes[1]
    if task in ['viral', 'multi']:
        ax0.scatter(dev_loss_horz, dev_losses_v, label='viral')
        ax1.scatter(dev_f1_horz, dev_f1_scores_v, label='viral') 
    if task in ['stance','multi']:
        ax0.scatter(dev_loss_horz, dev_losses_s, label='stance')
        ax1.scatter(dev_f1_horz, dev_f1_scores_s, label='stance')
    
    ax0.scatter(dev_loss_horz, dev_losses, label='obj')
    ax1.scatter(dev_f1_horz, dev_f1_scores, label='obj')
    
    #if task in ['viral','multi']: ax0.scatter(dev_loss_horz, dev_losses_v, label='viral')
    #if task in ['stance','multi']: ax0.scatter(dev_loss_horz, dev_losses_s, label='stance')
    #if task=='multi': ax0.scatter(dev_loss_horz, dev_losses, label='multi')
    #ax0.scatter(dev_loss_horz, dev_losses)
    ax0.set_ylabel('Training, dev losses')
    ax0.set_xlabel('Minibatch')
    ax0.legend()
    ax0.grid(True)
    
    #if task in ['viral','multi']: ax1.scatter(dev_f1_horz, dev_f1_scores_v, label='viral') 
    #if task in ['stance','multi']: ax1.scatter(dev_f1_horz, dev_f1_scores_s, label='stance')
    #if task=='multi': ax1.scatter(dev_f1_horz, dev_f1_scores, label='multi')
    ax1.legend()
    ax1.set_ylabel('Dev F1 score')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    plt.tight_layout()
    time.sleep(1)
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
    
    y_true_s = None
    y_pred_s = None
    all_logits_s = None
    y_true_v = None
    y_pred_v = None
    all_logits_v = None
    
    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
            if batch_id % log_interval == 0:
                logger.info(('\tTesting '+print_string+' Minibatch: %4d' % batch_id))
            
            #x0 = minibatch[0].to(gpu)  # index in orig data (unused)
            x1 = minibatch[1].to(gpu)   # encoded tokens
            x2 = minibatch[2].to(gpu)   # token_type_ids 
            x3 = minibatch[3].to(gpu)   # attention_mask 
            #x4 = minibatch[4].to(gpu)  # times_labeled (unused)
            #y = minibatch[5].to(gpu)   # true label 6 stance class (unused)
            y_s = minibatch[6].to(gpu)  # true label 4 stance class
            y_v = minibatch[7].to(gpu)  # viral_score
            
            outputs = model(input_ids=x1,    # shape=(n,C) where n=batch size
                            attention_mask=x3, 
                            token_type_ids=x2,
                            task='multi')   
            logits_s = outputs[0]
            logits_v = outputs[1]
            
            if y_true_s is None:                # for handling 1st minibatch
                y_true_s = y_s.clone().to(cpu)  # shape=(n,)
                y_true_v = y_v.clone().to(cpu)  # shape=(n,)
                idx_s = logits_s.argmax(1)      # for finding index of max value
                idx_v = logits_v.argmax(1)      # for finding index of max value
                y_pred_s = idx_s.clone().to(cpu)
                y_pred_v = idx_v.clone().to(cpu)
                all_logits_s = logits_s.clone().to(cpu)
                all_logits_v = logits_v.clone().to(cpu)
            else:                               # for all other minibatches
                y_true_s = torch.cat((y_true_s, y_s.clone().to(cpu)), 0)
                y_true_v = torch.cat((y_true_v, y_v.clone().to(cpu)), 0)
                idx_s = logits_s.argmax(1)      # for finding index of max value
                idx_v = logits_v.argmax(1)      # for finding index of max value
                y_pred_s = torch.cat((y_pred_s, idx_s.clone().to(cpu)), 0)
                y_pred_v = torch.cat((y_pred_v, idx_v.clone().to(cpu)), 0)
                all_logits_s = torch.cat((all_logits_s, logits_s.clone().to(cpu)), 0)
                all_logits_v = torch.cat((all_logits_v, logits_v.clone().to(cpu)), 0)
    return [y_pred_s, y_pred_v, y_true_s, y_true_v, all_logits_s, all_logits_v] # all have shape of (n,) except logits (n, num_class)

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
    
def f1_metrics_msg_stance(precisions, recalls, f1scores, supports, accuracy):
    macro_f1_score = sum(f1scores) / len(f1scores)
    weighted_f1_score = np.sum(f1scores * supports) / supports.sum()
    string = '\n============ Predict Tweet Stance ============\n'
    string +='Labels \tPrec. \tRecall\tF1    \tSupp  \n'
    string +='Denial \t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[0], recalls[0], f1scores[0], supports[0])
    string +='Support\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[1], recalls[1], f1scores[1], supports[1])
    string +='Comment\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[2], recalls[2], f1scores[2], supports[2])
    string +='Queries\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[3], recalls[3], f1scores[3], supports[3])
    string +='MacroF1\t%1.4f\n' % macro_f1_score
    string +='F1w_avg\t%1.4f\n' % weighted_f1_score
    string +='Acc    \t%2.1f\n' % (accuracy * 100)
    return string

def f1_metrics_msg_viral(precisions, recalls, f1scores, supports, accuracy):
    macro_f1_score = sum(f1scores) / len(f1scores)
    weighted_f1_score = np.sum(f1scores * supports) / supports.sum()
    string = '\n============ Predict Viral Tweets ============\n'
    string +='Labels \tPrec. \tRecall\tF1    \tSupp  \n'
    string +='N_viral\t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[0], recalls[0], f1scores[0], supports[0])
    string +='Viral  \t%1.4f\t%1.4f\t%1.4f\t%d\n' % (precisions[1], recalls[1], f1scores[1], supports[1])
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
        print(string, flush=True)
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
    ''' ============= Generic arguments first =============='''
    parser.add_argument("--batch_train",    default=1, type=int, help="minibatch size for training")
    parser.add_argument("--batch_test",     default=1, type=int, help="minibatch size for testing")
    parser.add_argument("--epochs",         default=1, type=int, help="maximum num of training epochs")
    parser.add_argument("--learning_rate",  default=1, type=float,help="learning rate")
    parser.add_argument("--optimizer",      default="adam",      help="adam or rmsprop")
    
    parser.add_argument("--do_train",       action="store_true", help="Whether to run training")
    parser.add_argument("--do_test",        action="store_true", help="Whether to run tests")
    
    parser.add_argument("--model_name",     default="my_modelA0",help="model name")
    parser.add_argument("--exp_name",       default="expXX",     help="Log filename prefix")
    
    parser.add_argument("--train_data",     default='./data/train_set_128_w_length_bertweet.bin')
    parser.add_argument("--test_data",      default='./data/test_set_128_w_length_bertweet.bin')
    
    parser.add_argument("--k_folds",        default=4, type=int, help='number of segments to fold training data')
    parser.add_argument("--folds2run",      default=1, type=int, help='number of times to do validation folding')
    
    parser.add_argument("--debug",          action="store_true", help="Debug flag")
    parser.add_argument("--log_interval",   default=1, type=int, help="num of batches before printing")
    ''' ===================================================='''
    ''' ========== Add additional arguments here ==========='''
    parser.add_argument('--loss_fn',        default='ce_loss',   help='loss function. ce_loss (default) or dice')
    parser.add_argument('--w_sample',       action='store_true', help='non flat sampling of training examples')
    parser.add_argument('--pretrain_model', default='',          help='model file that was pretrained on big twitter dataset')
    parser.add_argument('--epochs2giveup',  default=5, type=int, help='training is stopped if no improvements are seen after this number of epochs')
    parser.add_argument('--dropout',        default=0.1,type=float, help='dropout probability of last layer')
    parser.add_argument('--viral_threshold',default=80, type=float, help='percentile to define viral post')
    parser.add_argument('--weight_attr',    default='stance',    help='attribute for weighted sampling. must be "stance", "likes", "retweets"')
    parser.add_argument('--task',           default='multi',     help='task to train on. must be "multi", "stance", "viral"')
    parser.add_argument('--mtt_weight',     default=1.0,type=float, help='relative weight of viral task to stance task')
    ''' ===================================================='''
    return parser.parse_args()


if __name__ == '__main__':
    main()