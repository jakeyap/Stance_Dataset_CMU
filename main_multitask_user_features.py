#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:43:45 2021
this is similar to the multitask training file "main_multitasking.py"
the difference is that the tasks are trained sequentially, with the option to use different dataloaders. 
the sampling of training data can be done separately. 

prediction of virality wouldnt use features of the reply to gauge

For handling stance and virality multi task training
    
For the TRAINING SET, if you want to split virality based on LIKES, the thresholds are as follows
10% : 5
20% : 23
30% : 58
50% : 303
70% : 1424.9
80% : 3491
90% : 11453

For the TRAINING SET, if you want to split virality based on RETWEETS, the thresholds are as follows
10% : 2
20% : 8
30% : 26
50% : 131
70% : 589.9
80% : 1412
90% : 4257

@author: jakeyap
"""

import dataloader_utilities as dataloader
import tokenizer_v2
from tokenizer_v5 import tokenizer

from classifier_models import my_Bertweet, mtt_Bertweet, mtt_Bertweet2, SelfAdjDiceLoss
# from transformers import BertConfig

# default imports
import torch
import torch.optim as optim
import numpy as np

import logging, sys, argparse
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as f1_help
from sklearn.metrics import confusion_matrix

# insert at 1, 0 is the script path (or '' in REPL)
import os
lib_folder = '~/Projects/yk_python_lib'
lib_folder = os.path.expanduser(lib_folder)
sys.path.insert(1, lib_folder)
from misc_helpers import fmt_time_pretty

# DONE: implement kfolding
# DONE: ensure kfold prints properly
# DONE: edit the dataloader df2dl_v4
# DONE: create new model to parse dual tweets, meta data
# DONE: edit the train function for mtt
# DONE: edit the test function mtt
# DONE: change get model
# DONE: run single training example for sanity checking
# DONE: debug code segments

torch.manual_seed(0)

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
    LAYERS =        args.layers
    V_ATTR =        args.viral_attr
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
        
    logger.info('--------------- Getting dataframes -----------------')
    test_df = torch.load(TEST_DATA)
    full_train_df = torch.load(TRAIN_DATA)
    
    if DEBUG:
        test_df = test_df[0:40]
        full_train_df = test_df
    
    test_dl = dataloader.df_2_dl_v4(test_df, 
                                    batch_size=TEST_MB_SIZE, 
                                    randomize=False,
                                    viral_attr=V_ATTR,
                                    viral_threshold=V_THRESHOLD, 
                                    logger=logger)
    TESTLENGTH = len(test_df)
    
    if DO_TRAIN:
        logger.info('-------------- Setting loss function  --------------')
        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        if LOSS_FN == 'dice':
            loss_fn_s = SelfAdjDiceLoss(reduction='mean')           # for stance
            loss_fn_v = SelfAdjDiceLoss(reduction='mean')           # for viral
        elif LOSS_FN == 'ce_loss':
            logger.info('chose ce_loss')
            loss_fn_s = torch.nn.CrossEntropyLoss(reduction='mean') # for stance
            loss_fn_v = torch.nn.CrossEntropyLoss(reduction='mean') # for viral
        elif LOSS_FN == 'w_ce_loss':            
            logger.info('chose w_ce_loss')
            # count number of examples per category for stance
            stance_counts = torch.tensor([.1, .1, .1, .1])          # memory for storing counts
            for stance in full_train_df.number_labels_4_types:      # for each label type
                stance_counts [stance] += 1                         # count occurences
            stance_weights = 1.0 / stance_counts                    # inverse counts to get weights
            stance_weights = stance_weights / stance_weights.mean() # normalize so mean is 1
            
            # dont need to count number of examples per category for viral, defined by 80-20 split
            viral_weights = torch.tensor([0.1, 0.1])                # memory for weights
            viral_weights[0] = 100 / V_THRESHOLD                    # weight for not viral
            viral_weights[1] = 100 / (100 - V_THRESHOLD)            # weight for viral
            
            viral_weights = viral_weights / viral_weights.mean()    # normalize so mean is 1
            
            logger.info('stance loss weights')
            logger.info(stance_weights)
            logger.info('viral loss weights')
            logger.info(viral_weights)
            loss_fn_s = torch.nn.CrossEntropyLoss(reduction='mean', # loss function for stance
                                                  weight=stance_weights.cuda()) 
            loss_fn_v = torch.nn.CrossEntropyLoss(reduction='mean', # loss function for viral
                                                  weight=viral_weights.cuda())
        else:
            raise Exception('Loss function not found: ' + LOSS_FN)
        
        kfold_helper = KFold(n_splits=KFOLDS)
        kfolds_ran = 0
        kfolds_devs = []
        kfolds_tests= []
        '''
        logger.info('--------------- Getting fresh model ----------------')
        model = get_model(logger,MODEL_NAME, DROPOUT, LAYERS)
        model.cuda()
        model = torch.nn.DataParallel(model)
        if PRETRAIN != '':  # reload pretrained model 
            logger.info('loading pretrained model file ' + PRETRAIN)
            saved_params = torch.load(PRETRAIN)
            model.load_state_dict(saved_params)
            del saved_params
        
        ORIG_STATE_DICT = model.state_dict()    # hold a master starting copy of parameters
        '''
        for train_idx, dev_idx in kfold_helper.split(full_train_df):
            logger.info('--------------- Running KFOLD %d / %d ----------------' % (kfolds_ran+1, KFOLDS))
            print_gpu_obj()
            if FOLDS2RUN == 0:  # for debugging purposes
                train_df = full_train_df
                dev_df = full_train_df
            else:
                train_df = full_train_df.iloc[train_idx]
                dev_df = full_train_df.iloc[dev_idx]
            
            logger.info('------------ Converting to dataloaders -------------')
            train_dl = dataloader.df_2_dl_v4(train_df, 
                                             batch_size=TRNG_MB_SIZE, 
                                             randomize=True, 
                                             weighted_sample=W_SAMPLE, 
                                             weight_attr=W_ATTR,
                                             viral_attr=V_ATTR,
                                             viral_threshold=V_THRESHOLD, 
                                             logger=logger)
            dev_dl = dataloader.df_2_dl_v4(dev_df, 
                                           batch_size=TEST_MB_SIZE, 
                                           randomize=False, 
                                           weighted_sample=False,
                                           viral_attr=V_ATTR,
                                           viral_threshold=V_THRESHOLD, 
                                           logger=logger)
            
            logger.info('--------------- Getting fresh model ----------------')
            model = get_model(logger,MODEL_NAME, DROPOUT, LAYERS)
            model.cuda()
            model = torch.nn.DataParallel(model)
            if PRETRAIN != '':  # reload pretrained model 
                logger.info('loading pretrained model file ' + PRETRAIN)
                saved_params = torch.load(PRETRAIN)
                model.load_state_dict(saved_params)
                del saved_params
            
            logger.info('-------- Reload orig parameters into model ---------')
            #model.load_state_dict(ORIG_STATE_DICT)
            
            logger.info('----------------- Setting optimizer ----------------')
            if OPTIM=='adam':
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
            else:
                raise Exception('Optimizer not found: ' + optimizer)
            
            logger.info('------ Running a random test before training -------')
            _, _, _, _, random_test_idx = test_single_example(model=model, 
                                                              datalen=TESTLENGTH, 
                                                              dataloader=test_dl, 
                                                              logger=logger, 
                                                              log_interval=LOG_INTERVAL, 
                                                              index=-1, show=True)
            
            logger.info('---------------- Starting training -----------------')
            plotfile_fold = plotfile.replace('.png', '_fold'+str(kfolds_ran)+'.png')
            model_savefile_fold = model_savefile.replace('.bin', '_fold'+str(kfolds_ran)+'.bin')
            fold_metrics = train(model=model, train_dl=train_dl, dev_dl=dev_dl, 
                                 logger=logger, log_interval=LOG_INTERVAL, epochs=EPOCHS,
                                 loss_fn_s=loss_fn_s, loss_fn_v=loss_fn_v, optimizer=optimizer, 
                                 plotfile=plotfile_fold, modelfile=model_savefile_fold,
                                 epochs_giveup=EPOCHS2GIVEUP,
                                 task=TASK, mtt_weight=MTT_WEIGHT)
            
            kfolds_devs.append(fold_metrics)
            
            # reload best models
            saved_params = torch.load(model_savefile_fold)
            model.load_state_dict(saved_params)
            del saved_params # this is a huge memory sucker
            
            logger.info('------ Running same random test post training ------')
            test_single_example(model=model, 
                                datalen=TESTLENGTH, 
                                dataloader=test_dl, 
                                logger=logger, 
                                log_interval=LOG_INTERVAL, 
                                index=random_test_idx, 
                                show=True)
            
            logger.info('------- Running on test set after training  --------')
            test_results = test(model=model, 
                                dataloader=test_dl,
                                logger=logger,
                                log_interval=LOG_INTERVAL,
                                print_string='test')
            
            y_pred_s = test_results[0]
            y_pred_v = test_results[1]
            y_true_s = test_results[2]
            y_true_v = test_results[3]
            
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
            kfolds_tests.append([f1_metrics_s, f1_metrics_v, acc_s, acc_v, msg_s+msg_v])
            time2 = time.time()
            logger.info(fmt_time_pretty(time1, time2))
            
            del optimizer, model
            torch.cuda.empty_cache()
            
            kfolds_ran += 1
            if kfolds_ran >= FOLDS2RUN:
                break
        # finished kfolds, print everything once more, calculate the average f1 metrics
        f1s_s = []  # to accumulate stance f1 scores
        f1s_v = []  # to accumulate viral f1 scores
        accs_s = [] # to accumulate stance accuracy scores
        accs_v = [] # to accumulate viral accuracy scores
        for i in range(len(kfolds_devs)):
            fold_dev_results = kfolds_devs[i]
            fold_test_results = kfolds_tests[i]
            dev_msg = fold_dev_results[-1]
            test_msg = fold_test_results[-1]
            msg_2_print =               '\n******************** Fold %d results ********************\n' % i 
            msg_2_print = msg_2_print + '------------------------ Dev set ------------------------' + dev_msg 
            msg_2_print = msg_2_print + '------------------------ Test set ------------------------' + test_msg
            logger.info(msg_2_print)
            
            f1_s_metrics = fold_test_results[0]
            f1_v_metrics = fold_test_results[1]
            f1_s = np.average(f1_s_metrics [2]) # get individual class f1 scores, then avg
            f1_v = np.average(f1_v_metrics [2]) # get individual class f1 scores, then avg
            f1s_s.append(f1_s)                  # store macro f1 
            f1s_v.append(f1_v)                  # store macro f1 
            
            acc_s = fold_test_results[2]
            acc_v = fold_test_results[3]
            accs_s.append(acc_s)
            accs_v.append(acc_v)
        
        f1_s_avg = np.average(f1s_s)
        f1_s_std = np.std(f1s_s)
        f1_v_avg = np.average(f1s_v)
        f1_v_std = np.std(f1s_v)
        acc_s_avg= np.average(accs_s)
        acc_v_avg= np.average(accs_v)
        
        msg = '\nPerf across folds\n'
        msg+= 'avg_f1_stance\t%.4f\n' % f1_s_avg
        msg+= 'std_f1_stance\t%.4f\n' % f1_s_std
        msg+= 'avg_f1_viral\t%.4f\n' % f1_v_avg
        msg+= 'std_f1_viral\t%.4f\n' % f1_v_std
        msg+= 'avg acc_stance\t%.4f\n' % acc_s_avg
        msg+= 'avg_acc_viral\t%.4f\n' % acc_v_avg
        logger.info(msg)
        
    if DO_TEST:
        logger.info('------------------ Getting model -------------------')
        model = get_model(logger,MODEL_NAME, DROPOUT, LAYERS)
        model.cuda()
        model = torch.nn.DataParallel(model)
        if PRETRAIN != '':  # reload pretrained model 
            logger.info('loading pretrained model file ' + PRETRAIN)
            saved_params = torch.load(PRETRAIN)
            model.load_state_dict(saved_params)
        
        results = test(model=model, 
                       dataloader=test_dl,
                       logger=logger,
                       log_interval=LOG_INTERVAL,
                       print_string='test')
        
        y_pred_s = results[0]
        y_pred_v = results[1]
        y_true_s = results[2]
        y_true_v = results[3]
        
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
    time2 = time.time()
    logger.info(fmt_time_pretty(time1, time2))
    return

def get_model(logger=None, modelname='', dropout=0.1, layers=2):
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
    elif modelname=='mtt_Bertweet2':
        model = mtt_Bertweet2(4, dropout, layers)
    else:
        msg = 'model not found, exiting ' + modelname
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        raise Exception
    
    return model

def train(model, train_dl, dev_dl, logger, log_interval, epochs, loss_fn_s, loss_fn_v, optimizer, plotfile, modelfile, epochs_giveup=10, task='multi', mtt_weight=1.0):
    '''
    all the params needed are straightforward. except for plotfile, modelfile, ep
    model :         pytorch neural network model
        self explanatory
    train_dl :      training dataloader
        self explanatory
    dev_dl :        dev set dataloader
        self explanatory
    logger :        python logger
        self explanatory
    log_interval :  int
        how many epochs before printing progress
    epochs :        int
        max number of epochs to run
    loss_fn_s :     pytorch loss function
        loss function for stance
    loss_fn_v :     pytorch loss function
        loss function for viral
    optimizer :     pytorch optimizer
        self explanatory
    plotfile :      string
        filename to save plot to
    modelfile :     string
        filename to save model params to
    epochs_giveup : int
        if this number of epochs pass w/o any improvements to f1 score, give up. 
    task :          string
        what task to train on. "multi", "stance" or "viral"
    mtt_weight :    float
        relative weight of viral : stance loss. defaults to 1
    
    Returns
    -------
    f1 metric stance :  tuple
        precisions[0:3], recalls[0:3], f1[0:3], supports[0:3]
    f1 metric viral :   tuple
        precisions[0:1], recalls[0:1], f1[0:1], supports[0:1]
    accuracy stance :   float
        self explanatory
    accuracy viral :    float
        self explanatory
    message to print :  a string to print later
        self explanatory

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
    cpu = torch.device("cpu")
    for epoch in range(epochs):
        model.train()   # set model into training mode
        for batch_id, minibatch in enumerate(train_dl):
            if batch_id % log_interval == 0:
                logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch, batch_id)))

            #x0 = minibatch[0].to(gpu)  # index in orig data (unused)
            x1 = minibatch[1].to(gpu)   # encoded_tweets_h
            x2 = minibatch[2].to(gpu)   # token_type_ids_h 
            x3 = minibatch[3].to(gpu)   # attention_mask_h
            x4 = minibatch[4].to(gpu)   # encoded_tweets_t
            x5 = minibatch[5].to(gpu)   # token_type_ids_t 
            x6 = minibatch[6].to(gpu)   # attention_mask_t
            x7 = minibatch[7].float()   # followers_head
            x8 = minibatch[8].float()   # followers_tail
            x9 = minibatch[9].float()   # interaction_type_num
            x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
            x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
            x9 = x9.to(gpu)
            y_s =minibatch[10].to(gpu)  # true label 4 stance class
            y_v =minibatch[11].to(gpu)  # viral_score
            #print(x7.dtype)
            #print(x8.dtype)
            #print(x9.dtype)
            
            outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                            input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                            followers_head=x7, followers_tail=x8, int_type_num=x9,
                            task=task)
            logits_s = outputs[0]
            logits_v = outputs[1]
            
            if task=='stance':
                loss_v = 0
                loss_s = loss_fn_s(logits_s, y_s)   # calculate the stance loss
                losses_s.append(loss_s.item())      # archive the loss
                loss = loss_s
            elif task=='viral':
                loss_s = 0
                loss_v = loss_fn_v(logits_v, y_v)   # calculate the viral loss
                losses_v.append(loss_v.item())      # archive the loss
                loss = loss_v
            elif task=='multi':
                loss_s = loss_fn_s(logits_s, y_s)   # calculate the stance loss
                losses_s.append(loss_s.item())      # archive the loss
                loss_v = loss_fn_v(logits_v, y_v)   # calculate the viral loss
                losses_v.append(loss_v.item())      # archive the loss
                loss = loss_s+mtt_weight*loss_v     # sum the losses
                loss = loss / (1 + mtt_weight)
            else:
                err_string = 'task not found : ' + task
                logger.info(err_string)
                raise Exception(err_string)
                
            loss.backward()             # backward prop
            optimizer.step()            # step the gradients once
            optimizer.zero_grad()       # clear gradients before next step
            loss_value = loss.item()    # get value of total loss
            losses.append(loss_value)   # archive the total loss
            
            del x1,x2,x3,x4,x5,x6,x7,x8,x9, y_s, y_v
            del loss, outputs, logits_s, logits_v, loss_s, loss_v
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
        
        dev_loss_s = loss_fn_s(logits_s.to(gpu), y_true_s.to(gpu))
        dev_loss_v = loss_fn_v(logits_v.to(gpu), y_true_v.to(gpu))
        #dev_loss_s = loss_fn_s(logits_s, y_true_s)
        #dev_loss_v = loss_fn_v(logits_v, y_true_v)
        dev_loss_value_s = dev_loss_s.to(cpu).item()
        dev_loss_value_v = dev_loss_v.to(cpu).item()
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
        ax0.scatter(dev_loss_horz, dev_losses_v, label='viral_dev')
        ax1.scatter(dev_f1_horz, dev_f1_scores_v, label='viral_dev') 
    if task in ['stance','multi']:
        ax0.scatter(dev_loss_horz, dev_losses_s, label='stance_dev')
        ax1.scatter(dev_f1_horz, dev_f1_scores_s, label='stance_dev')
    
    ax0.scatter(dev_loss_horz, dev_losses, label='dev_loss')
    ax0.scatter(loss_horz, losses, label='train_loss')
    ax1.scatter(dev_f1_horz, dev_f1_scores, label='obj')
    
    #if task in ['viral','multi']: ax0.scatter(dev_loss_horz, dev_losses_v, label='viral')
    #if task in ['stance','multi']: ax0.scatter(dev_loss_horz, dev_losses_s, label='stance')
    #if task=='multi': ax0.scatter(dev_loss_horz, dev_losses, label='multi')
    #ax0.scatter(dev_loss_horz, dev_losses)
    ax0.set_ylabel('Training, dev losses')
    ax0.set_xlabel('Minibatch')
    ax0.legend()
    ax0.grid(True)
    ax0.set_yscale('log')
    
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
    
    return [f1_metrics_s, f1_metrics_v, acc_s, acc_v, msg_s + msg_v]

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
            x1 = minibatch[1].to(gpu)   # encoded_tweets_h
            x2 = minibatch[2].to(gpu)   # token_type_ids_h 
            x3 = minibatch[3].to(gpu)   # attention_mask_h
            x4 = minibatch[4].to(gpu)   # encoded_tweets_t
            x5 = minibatch[5].to(gpu)   # token_type_ids_t 
            x6 = minibatch[6].to(gpu)   # attention_mask_t
            x7 = minibatch[7].float()   # followers_head
            x8 = minibatch[8].float()   # followers_tail
            x9 = minibatch[9].float()   # interaction_type_num
            x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
            x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
            x9 = x9.to(gpu)
            
            y_s =minibatch[10].to(gpu)  # true label 4 stance class
            y_v =minibatch[11].to(gpu)  # viral_score
            
            outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                            input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                            followers_head=x7, followers_tail=x8, int_type_num=x9,
                            task='multi')
            logits_s = outputs[0]
            logits_v = outputs[1]
            
            if y_true_s is None:                            # for handling 1st minibatch
                y_true_s = y_s.clone().to(cpu)     # shape=(n,)
                y_true_v = y_v.clone().to(cpu)     # shape=(n,)
                idx_s = logits_s.argmax(1)                  # for finding index of max value for stance
                idx_v = logits_v.argmax(1)                  # for finding index of max value for viral
                y_pred_s = idx_s.clone().to(cpu)   # copy the index to cpu
                y_pred_v = idx_v.clone().to(cpu)   # copy the index to cpu
                all_logits_s = logits_s.clone().to(cpu)    # copy the stance logits
                all_logits_v = logits_v.clone().to(cpu)    # copy the viral logits
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

def test_single_example(model, datalen, dataloader, logger, log_interval, index=-1, show=True):
    model.eval()
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    if index==-1:   # generate a random number
        index = np.random.randint(0, datalen)
    logger.info(index)
    batchsize = dataloader.batch_size    
    inter_batch_idx = index // batchsize 
    intra_batch_idx = index % batchsize

    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
            logger.info('batch id %d ' % batch_id)
            if inter_batch_idx == batch_id:
                #x0 = minibatch[0].to(gpu)  # index in orig data (unused)
                x1 = minibatch[1].to(gpu)   # encoded_tweets_h
                x2 = minibatch[2].to(gpu)   # token_type_ids_h 
                x3 = minibatch[3].to(gpu)   # attention_mask_h
                x4 = minibatch[4].to(gpu)   # encoded_tweets_t
                x5 = minibatch[5].to(gpu)   # token_type_ids_t 
                x6 = minibatch[6].to(gpu)   # attention_mask_t
                x7 = minibatch[7].float()   # followers_head
                x8 = minibatch[8].float()   # followers_tail
                x9 = minibatch[9].float()   # interaction_type_num
                x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
                x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
                x9 = x9.to(gpu)
                
                y_true_s = minibatch[10]    # true label 4 stance class
                y_true_v = minibatch[11]    # viral_score
                
                outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                            input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                            followers_head=x7, followers_tail=x8, int_type_num=x9,
                            task='multi')
                
                logits_s = outputs[0].to(cpu)
                logits_v = outputs[1].to(cpu)
                
                y_pred_s = logits_s.argmax(1)
                y_pred_v = logits_v.argmax(1)
                
                y_true_s = y_true_s[intra_batch_idx].item()
                y_pred_s = y_pred_s[intra_batch_idx].item()
                y_true_v = y_true_v[intra_batch_idx].item()
                y_pred_v = y_pred_v[intra_batch_idx].item()
                
                if show:
                    text_head = x1.to(cpu)[intra_batch_idx]             # get correct row of data for head
                    text_tail = x4.to(cpu)[intra_batch_idx]             # get correct row of data for tail
                    text_head = tokenizer.decode(text_head.tolist())    # convert head to list, then decode to text
                    text_tail = tokenizer.decode(text_tail.tolist())    # convert tail to list, then decode to text
                    text_head = text_head.replace(' <pad>', '')         # remove padding before printing
                    text_tail = text_tail.replace(' <pad>', '')         # remove padding before printing
                    logger.info('Original tweet index: ' + str(index))
                    logger.info('Original head tweet: ' + text_head)
                    logger.info('Original tail tweet: ' + text_tail)
                    logger.info('Original Stance Label:  ' + tokenizer_v2.convert_label_num2string(y_true_s,4))
                    logger.info('Predicted Stance Label: ' + tokenizer_v2.convert_label_num2string(y_pred_s,4))
                    logger.info('Original Viral Label:   ' + str(y_true_v))
                    logger.info('Predicted Viral Label:  ' + str(y_pred_v))
                
                return [y_true_s, y_pred_s, y_true_v, y_pred_v, index]
    
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
    
    parser.add_argument("--model_name",     default="mtt_Bertweet2", help="model name")
    parser.add_argument("--exp_name",       default="expXX",     help="Log filename prefix")
    
    parser.add_argument("--train_data",     default='./data/train_set_128_individual_bertweet.bin')
    parser.add_argument("--test_data",      default='./data/test_set_128_individual_bertweet.bin')
    
    parser.add_argument("--k_folds",        default=4, type=int, help='number of segments to fold training data')
    parser.add_argument("--folds2run",      default=1, type=int, help='number of times to do validation folding')
    
    parser.add_argument("--debug",          action="store_true", help="Debug flag")
    parser.add_argument("--log_interval",   default=1, type=int, help="num of batches before printing")
    ''' ===================================================='''
    ''' ========== Add additional arguments here ==========='''
    parser.add_argument('--loss_fn',        default='ce_loss',   help='loss function. ce_loss (default), dice, w_ce_loss')
    parser.add_argument('--w_sample',       action='store_true', help='non flat sampling of training examples')
    parser.add_argument('--pretrain_model', default='',          help='model file that was pretrained on big twitter dataset')
    parser.add_argument('--epochs2giveup',  default=5, type=int, help='training is stopped if no improvements are seen after this number of epochs')
    parser.add_argument('--dropout',        default=0.1,type=float, help='dropout probability of last layer')
    parser.add_argument('--layers',         default=2, type=int, help='number of level 2 transformer layers')
    parser.add_argument('--viral_attr',     default='likes',     help='what attribute to use to define viral, must be ["likes", "retweets"]')
    parser.add_argument('--viral_threshold',default=80, type=float, help='percentile to define viral post')
    parser.add_argument('--weight_attr',    default='stance',    help='attribute for weighted sampling. must be "stance", "viral"')
    parser.add_argument('--task',           default='multi',     help='task to train on. must be "multi", "stance", "viral"')
    parser.add_argument('--mtt_weight',     default=1.0,type=float, help='relative weight of viral task to stance task')
    ''' ===================================================='''
    return parser.parse_args()


def print_gpu_obj():
    import gc
    count = 0
    for tracked_obj in gc.get_objects():
        if torch.is_tensor(tracked_obj):
            if tracked_obj.is_cuda:
                count += 1
                #print('{} {} {}'.format(type(tracked_obj).__name__, 
                #                        " pinned" if tracked_obj.is_pinned() else "",
                #                        tracked_obj.shape))
    print('there are %d tracked objects in GPU' % count)

if __name__ == '__main__':
    main()