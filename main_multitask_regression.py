#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:24:53 2021

this is similar to the multitask training file "main_multitasking_user_features.py"
this uses the user keywords as part of the representation also

KEY CHANGES: 
    use regression instead of use 
    requires Prec@K, NDCG@K calculations
    
Definitions
Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
NDCG@K = DCG@K(predicted results) / DCG@K(perfect results)
    where DCG@K = sum_i_K [ relevance score of item i / log2(1 + rank of item i) ]

So in our context, 
Precision@K = (Num of true top K items) / (Num of top K items the model recommends)

prediction of virality wouldnt use features of the reply to gauge

For handling stance and virality multi task training

@author: jakeyap
"""

import dataloader_utilities as dataloader
import tokenizer_v2
from tokenizer_v5 import tokenizer

from classifier_models import SelfAdjDiceLoss
from classifier_models import mtt_Bertweet5, mtt_Bert5

# default imports
import torch
import torch.optim as optim
import numpy as np

import logging, sys, argparse
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as f1_help
from sklearn.metrics import r2_score, mean_squared_error, ndcg_score

# insert at 1, 0 is the script path (or '' in REPL)
import os
lib_folder = '~/Projects/yk_python_lib'
lib_folder = os.path.expanduser(lib_folder)
sys.path.insert(1, lib_folder)
from misc_helpers import fmt_time_pretty
import gc

# TODO: debug code segments

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
    V_LOG =         args.viral_log
    W_ATTR =        args.weight_attr 
    TASK =          args.task
    MTT_WEIGHT =    args.mtt_weight
    ABLATION =      args.ablation
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
    
    if V_ATTR == 'likes':
        viral_score = test_df.favorite_count
    elif V_ATTR =='retweets':
        viral_score = test_df.retweets_count
    else:
        raise Exception ('V_ATTR not found: '+ V_ATTR)
    
    top_percentiles = [10,20,30,40,50]              # percentiles to analyse
    for pctile in top_percentiles:
        thr = np.percentile(viral_score,            # get threshold
                            100-pctile)             # percentile function arg is CDF, so must minus 100
        top_ranked = (viral_score >= thr)           # label all posts as not viral
        string = 'top_ranked_'+str(pctile)          # column title 
        test_df[string] = top_ranked                # stick labels into dataframe
    
    test_dl = dataloader.df_2_dl_v6(test_df, 
                                    batch_size=TEST_MB_SIZE, 
                                    randomize=False,
                                    viral_attr=V_ATTR,
                                    logger=logger,
                                    ablation=ABLATION)
    
    TESTLENGTH = len(test_df)
    if DO_TRAIN:
        logger.info('-------------- Setting loss function  --------------')
        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        if LOSS_FN == 'dice':
            logger.info('chose dice')
            loss_fn_s = SelfAdjDiceLoss(reduction='mean')           # for stance
        elif LOSS_FN == 'ce_loss':
            logger.info('chose ce_loss')
            loss_fn_s = torch.nn.CrossEntropyLoss(reduction='mean') # for stance
        elif LOSS_FN == 'w_ce_loss':            
            logger.info('chose w_ce_loss')
            # count number of examples per category for stance
            stance_counts = torch.tensor([.1, .1, .1, .1])          # memory for storing counts
            for stance in full_train_df.number_labels_4_types:      # for each label type
                stance_counts [stance] += 1                         # count occurences
            stance_weights = 1.0 / stance_counts                    # inverse counts to get weights
            stance_weights = stance_weights / stance_weights.mean() # normalize so mean is 1
            
            logger.info('stance loss weights')
            logger.info(stance_weights)
            
            loss_fn_s = torch.nn.CrossEntropyLoss(reduction='mean', # loss function for stance
                                                  weight=stance_weights.cuda()) 
        else:
            raise Exception('Loss function not found: ' + LOSS_FN)
        
        loss_fn_v = torch.nn.MSELoss(reduction='mean')
        
        kfold_helper = KFold(n_splits=KFOLDS)
        kfolds_ran = 0
        kfolds_devs = []
        kfolds_tests= []
        
        for train_idx, dev_idx in kfold_helper.split(full_train_df):
            logger.info('--------------- Running KFOLD %d / %d ----------------' % (kfolds_ran+1, KFOLDS))
            logger.info(print_gpu_obj())
            if FOLDS2RUN == 0:  # for debugging purposes
                train_df = full_train_df
                dev_df = full_train_df
            else:
                train_df = full_train_df.iloc[train_idx]
                dev_df = full_train_df.iloc[dev_idx]
            
            logger.info('------------ Converting to dataloaders -------------')
            train_dl = dataloader.df_2_dl_v6(train_df, 
                                             batch_size=TRNG_MB_SIZE, 
                                             randomize=True, 
                                             weighted_sample=W_SAMPLE, 
                                             weight_attr=W_ATTR,
                                             viral_attr=V_ATTR,
                                             logger=logger,
                                             ablation=ABLATION)
            dev_dl = dataloader.df_2_dl_v6(dev_df, 
                                           batch_size=TEST_MB_SIZE, 
                                           randomize=False, 
                                           weighted_sample=False,
                                           viral_attr=V_ATTR,
                                           logger=logger,
                                           ablation=ABLATION)
            
            logger.info('--------------- Getting fresh model ----------------')
            model = get_model(logger,MODEL_NAME, DROPOUT, LAYERS)
            model.cuda()
            model = torch.nn.DataParallel(model)
            if PRETRAIN != '':  # reload pretrained model 
                logger.info('loading pretrained model file ' + PRETRAIN)
                saved_params = torch.load(PRETRAIN)
                model.load_state_dict(saved_params)
                del saved_params
            
            logger.info('----------------- Setting optimizer ----------------')
            if OPTIM=='adam':
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            else:
                raise Exception('Optimizer not found: ' + optimizer)
            
            logger.info('------ Running a random test before training -------')
            _, _, _, _, random_test_idx = test_single_example(model=model, 
                                                              datalen=TESTLENGTH, 
                                                              dataloader=test_dl, 
                                                              logger=logger, 
                                                              log_interval=LOG_INTERVAL, 
                                                              v_log=V_LOG,
                                                              index=-1, show=True)
            
            logger.info('---------------- Starting training -----------------')
            plotfile_fold = plotfile.replace('.png', '_fold'+str(kfolds_ran)+'.png')
            model_savefile_fold = model_savefile.replace('.bin', '_fold'+str(kfolds_ran)+'.bin')
            fold_metrics = train(model=model, train_dl=train_dl, dev_dl=dev_dl, 
                                 logger=logger, log_interval=LOG_INTERVAL, epochs=EPOCHS,
                                 loss_fn_s=loss_fn_s, loss_fn_v=loss_fn_v, optimizer=optimizer, 
                                 v_log=V_LOG, top_percentiles=top_percentiles, 
                                 plotfile=plotfile_fold, modelfile=model_savefile_fold,
                                 epochs_giveup=EPOCHS2GIVEUP,
                                 task=TASK, mtt_weight=MTT_WEIGHT)
            
            kfolds_devs.append(fold_metrics)
            
            # reload best models
            saved_params = torch.load(model_savefile_fold)
            model.load_state_dict(saved_params)
            del saved_params # this is a huge memory sucker
            with torch.no_grad(): # run some tests post training
                logger.info('------ Running same random test post training ------')
                test_single_example(model=model, 
                                    datalen=TESTLENGTH, 
                                    dataloader=test_dl, 
                                    logger=logger, 
                                    log_interval=LOG_INTERVAL,
                                    v_log=V_LOG,
                                    index=random_test_idx, 
                                    show=True)
                
                logger.info('------- Running on test set after training  --------')
                test_results = test(model=model, 
                                    dataloader=test_dl,
                                    logger=logger,
                                    log_interval=LOG_INTERVAL,
                                    v_log=V_LOG,
                                    print_string='test')
                
                y_pred_s = test_results[0]   # shape=(n,). elements are ints.
                y_pred_v = test_results[1]   # shape=(n,). elements are floats
                y_true_s = test_results[2]   # shape=(n,). elements are ints
                y_true_v = test_results[3]   # shape=(n,). elements are floats
                
                f1_metrics_s = f1_help(y_true_s, y_pred_s,  # calculate f1 scores for stance
                                       average=None,        # dont set to calculate for all
                                       labels=[0,1,2,3])    # number of classes = 4
                metrics_v = calc_rank_scores_at_k(y_true_v,
                                                  y_pred_v,
                                                  top_percentiles)
                
                prec_s, rec_s, f1s_s, supp_s = f1_metrics_s
                acc_s = calculate_acc(y_pred_s, y_true_s)
                msg_s = f1_metrics_msg_stance(prec_s, rec_s, f1s_s, supp_s, acc_s)
                
                prec_v, supp_v, ndcg_v = metrics_v
                r2e_v = r2_score(y_true_v, y_pred_v)
                mse_v = mean_squared_error(y_true_v, y_pred_v)
                msg_v = metrics_msg_viral(prec_v, supp_v, ndcg_v, top_percentiles, r2e_v, mse_v)
                
                logger.info(msg_s + msg_v)
                kfolds_tests.append([f1_metrics_s, acc_s, r2e_v, mse_v, msg_s+msg_v])
                time2 = time.time()
                logger.info(fmt_time_pretty(time1, time2))
            # ===================================================================
            # need to do these steps to force garbage collection to work properly
            # without it, the model deletion doesnt seem to work properly
            model.to('cpu') 
            del optimizer, model, train_dl, dev_dl
            gc.collect()
            torch.cuda.empty_cache()
            # ===================================================================
            
            kfolds_ran += 1
            if kfolds_ran >= FOLDS2RUN:
                break
        # finished kfolds, print everything once more, calculate the average f1 metrics
        f1s_s = []  # to accumulate stance f1 scores
        r2e_v = []  # to accumulate viral r2 scores
        mse_v = []  # to accumulate viral mse scores
        accs_s = [] # to accumulate stance accuracy scores
        
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
            acc_s = fold_test_results[1]            
            r2e_v = fold_test_results[2]
            mse_v = fold_test_results[3]
            
            f1_s = np.average(f1_s_metrics [2]) # get individual class f1 scores, then avg
            f1s_s.append(f1_s)                  # store macro f1 

            accs_s.append(acc_s)
        
        f1_s_avg = np.average(f1s_s)
        f1_s_std = np.std(f1s_s)
        r2_v_avg = np.average(r2e_v)
        r2_v_std = np.std(r2e_v)
        mse_v_avg = np.average(mse_v)
        mse_v_std = np.std(mse_v)
        acc_s_avg= np.average(accs_s)
        acc_s_std = np.std(accs_s)
        
        msg = '\nPerf across folds\n'
        msg+= 'avg_f1_stance\t%.4f\n' % f1_s_avg
        msg+= 'std_f1_stance\t%.4f\n' % f1_s_std
        msg+= 'avg_r2_viral\t%.4f\n' % r2_v_avg
        msg+= 'std_r2_viral\t%.4f\n' % r2_v_std
        msg+= 'avg_mse_viral\t%.4f\n' % mse_v_avg
        msg+= 'std_mse_viral\t%.4f\n' % mse_v_std
        msg+= 'avg_acc_stance\t%.4f\n' % acc_s_avg
        msg+= 'std_acc_stance\t%.4f\n' % acc_s_std
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
        
        test_results = test(model=model, 
                            dataloader=test_dl,
                            logger=logger,
                            log_interval=LOG_INTERVAL,
                            v_log=V_LOG,
                            print_string='test')
        
        y_pred_s = test_results[0]   # shape=(n,). elements are ints.
        y_pred_v = test_results[1]   # shape=(n,). elements are floats
        y_true_s = test_results[2]   # shape=(n,). elements are ints
        y_true_v = test_results[3]   # shape=(n,). elements are floats
        
        f1_metrics_s = f1_help(y_true_s, y_pred_s,  # calculate f1 scores for stance
                               average=None,        # dont set to calculate for all
                               labels=[0,1,2,3])    # number of classes = 4
        metrics_v = calc_rank_scores_at_k(y_true_v,
                                          y_pred_v,
                                          top_percentiles)
        
        prec_s, rec_s, f1s_s, supp_s = f1_metrics_s
        acc_s = calculate_acc(y_pred_s, y_true_s)
        msg_s = f1_metrics_msg_stance(prec_s, rec_s, f1s_s, supp_s, acc_s)
        
        prec_v, supp_v, ndcg_v = metrics_v        
        r2e_v = r2_score(y_true_v, y_pred_v)
        mse_v = mean_squared_error(y_true_v, y_pred_v)
        msg_v = metrics_msg_viral(prec_v, supp_v, ndcg_v, top_percentiles, r2e_v, mse_v)
        
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
    
    if modelname=='mtt_Bertweet5_regr':
        model = mtt_Bertweet5(4, dropout, layers, True)
    elif modelname=='mtt_Bert5_regr':
        model = mtt_Bert5(4, dropout, layers, True)
    else:
        msg = 'model not found, exiting ' + modelname
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        raise Exception
    
    return model

def train(model, train_dl, dev_dl, logger, log_interval, epochs, loss_fn_s, loss_fn_v, optimizer, v_log, plotfile, modelfile, top_percentiles, epochs_giveup=10, task='multi', mtt_weight=1.0):
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
    v_log :         boolean
        log the viral score if True.
    plotfile :      string
        filename to save plot to
    modelfile :     string
        filename to save model params to
    top_percentiles : list
        list of topK percentiles
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
    
    dev_mse_scores_v = []
    dev_r2e_scores_v = []
    dev_f1_scores_s = []
    dev_metric_scores = []
    dev_f1_horz = []
    best_metric = -1e9  # variable for deciding when to stop training
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
            x10= minibatch[10].to(gpu)  # user keywords
            
            x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
            x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
            x9 = x9.to(gpu)
            y_s =minibatch[11].to(gpu)  # true label 4 stance class
            y_v =minibatch[12].float().to(gpu)  # viral_score
            
            if v_log:
                y_v = torch.log10(y_v+0.1)      # to log viral score
            
            outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                            input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                            followers_head=x7, followers_tail=x8, int_type_num=x9, 
                            user_keywords=x10, task=task)
            logits_s = outputs[0]                   # shape=(n,4)
            logits_v = outputs[1]                   # shape=(n,1)
            
            if task=='stance':
                loss_v = 0
                loss_s = loss_fn_s(logits_s, y_s)   # calculate the stance loss
                losses_s.append(loss_s.item())      # archive the loss
                loss = loss_s
            elif task=='viral':
                loss_s = 0
                logits_v = logits_v.reshape(-1)     # shape=(n,)
                loss_v = loss_fn_v(logits_v, y_v)   # calculate the viral loss
                losses_v.append(loss_v.item())      # archive the loss
                loss = loss_v
            elif task=='multi':
                loss_s = loss_fn_s(logits_s, y_s)   # calculate the stance loss
                losses_s.append(loss_s.item())      # archive the loss
                logits_v = logits_v.reshape(-1)     # shape=(n,)
                loss_v = loss_fn_v(logits_v, y_v)   # calculate the viral loss
                losses_v.append(loss_v.item())      # archive the loss
                loss = loss_s+mtt_weight*loss_v     # weighted sum of losses
                loss = loss / (1 + mtt_weight)      # weighted sum of losses
            else:
                err_string = 'task not found : ' + task
                logger.info(err_string)
                raise Exception(err_string)
            
            loss.backward()             # backward prop
            optimizer.step()            # step the gradients once
            optimizer.zero_grad()       # clear gradients before next step
            loss_value = loss.item()    # get value of total loss
            losses.append(loss_value)   # archive the total loss
            # ===================================================================
            # not needed to actually free up memory, cauz the procedure exits
            # these variables are not returned, so not problematic
            # del x1,x2,x3,x4,x5,x6,x7,x8,x9, y_s, y_v
            # del loss, outputs, logits_s, logits_v, loss_s, loss_v
            # gc.collect()
            # ===================================================================
            
            if len(loss_horz)==0:
                loss_horz.append(0)
            else:
                loss_horz.append(len(loss_horz))
        model.eval()    # change back to eval mode
        results = test(model=model, 
                       dataloader=dev_dl,
                       logger=logger,
                       log_interval=log_interval,
                       v_log=v_log,
                       print_string='dev')
        
        y_pred_s = results[0]   # shape=(n,). elements are ints.
        y_pred_v = results[1]   # shape=(n,). elements are floats
        y_true_s = results[2]   # shape=(n,). elements are ints
        y_true_v = results[3]   # shape=(n,). elements are floats
        logits_s = results[4]   # shape=(n,4). elements are floats
        
        dev_loss_s = loss_fn_s(logits_s.to(gpu), y_true_s.to(gpu))
        dev_loss_v = loss_fn_v(y_true_v.to(gpu), y_pred_v.to(gpu)) # this was buggy just now
        
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
        
        prec_s, recall_s, f1s_s, supp_s = f1_metrics_s
        acc_s = calculate_acc(y_pred_s, y_true_s)
        msg_s = f1_metrics_msg_stance(prec_s, recall_s, f1s_s, supp_s, acc_s)
        
        metrics_v = calc_rank_scores_at_k(y_true_v,
                                          y_pred_v,
                                          top_percentiles)
        prec_v, supp_v, ndcg_v = metrics_v
        r2e_v = r2_score(y_true_v, y_pred_v)
        mse_v = dev_loss_value_v
        msg_v = metrics_msg_viral(prec_v, supp_v, ndcg_v, top_percentiles, r2e_v, mse_v)
        
        logger.info(msg_s + msg_v)
        
        f1_score_s = sum(f1s_s) / len(f1s_s)
        
        if task=='stance':
            curr_metric = f1_score_s
        elif task=='viral':
            curr_metric = r2e_v
        else:
            curr_metric = (f1_score_s + mtt_weight * r2e_v) / (1 + mtt_weight)
        
        dev_f1_scores_s.append(f1_score_s)
        dev_r2e_scores_v.append(r2e_v)
        dev_mse_scores_v.append(mse_v)
        
        dev_metric_scores.append(curr_metric)
        dev_f1_horz.append(epoch)
        epochs_since_best += 1
        
        if curr_metric > best_metric:       # if best metric score is reached
            logger.info('Best results so far. Saving model...')
            best_metric = curr_metric       # store best score
            epochs_since_best = 0           # reset the epochs counter
            torch.save(model.state_dict(), 
                       modelfile)   # save model
        
        if epochs_since_best >= epochs_giveup:
            logger.info('No improvements in F1 for %d epochs' % epochs_since_best)
            break                   # stop training if no improvements for too long
        
    state = torch.load(modelfile)   # reload best model
    model.load_state_dict(state)
    del state
    fig, axes = plt.subplots(3,1)
    fig.set_size_inches(10,8)
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    if task in ['viral', 'multi']:
        ax1.scatter(dev_f1_horz, dev_r2e_scores_v, label='v_dev_r2') 
        ax2.scatter(dev_f1_horz, dev_losses_v, label='v_dev_mse')
    if task in ['stance','multi']:
        ax0.scatter(dev_loss_horz, dev_losses_s, label='s_dev_loss')
        ax1.scatter(dev_f1_horz, dev_f1_scores_s, label='s_dev_f1')
    
    ax0.scatter(dev_loss_horz, dev_losses, label='dev_loss')
    ax0.scatter(loss_horz, losses, label='train_loss')
    
    ax0.set_ylabel('Train, dev losses')
    ax0.set_xlabel('Minibatch')
    ax0.legend()
    ax0.grid(True)
    ax0.set_yscale('log')
    
    ax1.legend()
    ax1.set_ylabel('Dev R2, F1s')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    ax2.legend()
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    time.sleep(1)
    fig.savefig(plotfile)
    
    return [f1_metrics_s, acc_s, r2e_v, mse_v, msg_s + msg_v]

def test(model, dataloader, logger, log_interval, v_log, print_string='test'):
    """
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
    v_log : boolean
        Log the viral scores if True
    print_string : string
        text to include into print lines. Default is 'test'
        
    Returns
    -------
    y_pred_s : linear tensor. shape=(n,). 
        Torch tensor of predicted stance class. Each element is an int
    y_pred_v : linear tensor. shape=(n,). 
        Torch tensor of predicted viral scores. Each element is a float
    
    y_true_s : linear tensor. shape=(n,).
        Torch tensor of true stance class. shape=(n,). Each element is an int
    y_true_v : linear tensor. shape=(n,).
        Torch tensor of true viral scores . shape=(n,). Each element is a float
    all_logits_s : tensor. shape=(n,4)
        Tensor of original output logits. Each element is a float
    """
    
    model.eval()
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    y_true_s = None
    y_pred_s = None
    all_logits_s = None
    y_true_v = None
    y_pred_v = None
    
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
            x10= minibatch[10].to(gpu)  # user keywords
            
            x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
            x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
            x9 = x9.to(gpu)
            
            y_s =minibatch[11].to(gpu)          # true label 4 stance class
            y_v =minibatch[12].float().to(gpu)  # viral_score
            
            if v_log:
                y_v = torch.log10(y_v+0.1)      # to log viral score
            
            outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                            input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                            followers_head=x7, followers_tail=x8, int_type_num=x9,
                            user_keywords=x10, task='multi')
            logits_s = outputs[0]
            logits_v = outputs[1].reshape(-1)
            
            # TODO delete later. for checking GPU reassemblying data
            #logits_v = minibatch[0].to(gpu)  # index in orig data (unused)
            #logger.info(logits_v)
            
            if y_true_s is None:                            # for handling 1st minibatch
                y_true_s = y_s.clone().to(cpu)      # shape=(n,)
                y_true_v = y_v.clone().to(cpu)      # shape=(n,)
                idx_s = logits_s.argmax(1)          # for finding index of max value for stance
                y_pred_s = idx_s.clone().to(cpu)    # copy the index to cpu
                y_pred_v = logits_v.clone().to(cpu) # copy the index to cpu
                all_logits_s = logits_s.clone().to(cpu)     # copy the stance logits
            else:                               # for all other minibatches
                y_true_s = torch.cat((y_true_s, y_s.clone().to(cpu)), 0)
                y_true_v = torch.cat((y_true_v, y_v.clone().to(cpu)), 0)
                idx_s = logits_s.argmax(1)      # for finding index of max value
                y_pred_s = torch.cat((y_pred_s, idx_s.clone().to(cpu)), 0)
                y_pred_v = torch.cat((y_pred_v, logits_v.clone().to(cpu)), 0)
                all_logits_s = torch.cat((all_logits_s, logits_s.clone().to(cpu)), 0)
            
    return [y_pred_s, y_pred_v, y_true_s, y_true_v, all_logits_s] # all have shape of (n,) except logits (n, num_class)

def test_single_example(model, datalen, dataloader, logger, log_interval, v_log, index=-1, show=True):
    model.eval()
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    if index==-1:   # generate a random number
        index = np.random.randint(0, datalen)
    
    batchsize = dataloader.batch_size    
    inter_batch_idx = index // batchsize 
    intra_batch_idx = index % batchsize

    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
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
                x10= minibatch[10].to(gpu)  # user keywords
                
                x7 = torch.log10(x7.to(gpu)+0.1)    # log to scale the numbers down to earth
                x8 = torch.log10(x8.to(gpu)+0.1)    # log to scale the numbers down to earth 
                x9 = x9.to(gpu)
                
                y_true_s = minibatch[11]    # true label 4 stance class
                y_true_v = minibatch[12]    # viral_score
                
                if v_log:
                    y_true_v = torch.log10(y_true_v+0.1)        # log the viral score
                
                outputs = model(input_ids_h=x1, token_type_ids_h=x2, attention_mask_h=x3,
                                input_ids_t=x4, token_type_ids_t=x5, attention_mask_t=x6, 
                                followers_head=x7, followers_tail=x8, int_type_num=x9,
                                user_keywords=x10, task='multi')
                
                logits_s = outputs[0].to(cpu)
                logits_v = outputs[1].to(cpu)
                
                y_pred_s = logits_s.argmax(1)
                y_pred_v = logits_v.reshape(-1)
                
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
                    logger.info('Original Viral Score:   ' + str(y_true_v))
                    logger.info('Predicted Viral Score:  ' + str(y_pred_v))
                
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

def metrics_msg_viral(precisions, supports, ndcgs, top_k_s, r2_err, m2_error):
    if len(precisions) != len(top_k_s):
        raise Exception('top_k_s and f1 metrics dimensions mismatch')
        
    string = '\n============ Predict Viral Tweets ============\n'
    string +='TopK\tPrec@K\t  Supp\tNDCG@K\n'
    for k in range(len(top_k_s)):
         string += '%2d  \t%1.4f\t%6d\t%1.4f\n' % (top_k_s[k], precisions[k], supports[k], ndcgs[k])
    
    string += 'MSE:\t%1.4f\t\n' % m2_error
    string += 'R2E:\t%1.4f\t\n' % r2_err
    
    return string

def calculate_acc(y_pred, y_true):
    correct = y_pred == y_true
    length = len(y_pred)
    return correct.sum().item() / length

def calc_rank_scores_at_k(y_true, y_pred, top_k_pctiles=[10,20]):
    # Given a list of scalar true and predicted scores, calculate F1 metrics for top K percentile elements.
    # Calculates precision@K, supports@K, ndcg@K
    
    precisions = []
    ndcg_scores = []
    supports = []
    
    for top_pctile in top_k_pctiles:
        pctile = 100 - top_pctile                   # top 10%-tile means 90%-tile CDF
        thr_true  = np.percentile(y_true, pctile)   # find threshold for true labels
        thr_pred  = np.percentile(y_pred, pctile)   # find threshold for predicted labels
        labels_true = y_true >= thr_true            # label +ve class in true_scores
        labels_pred = y_pred >= thr_pred            # label +ve class in predicted scores
        
        f1_metrics = f1_help(labels_true,           # calculate f1 for topK viral
                             labels_pred,           # binary classfiication
                             average='binary',
                             pos_label=1)
        
        num_top_rank = sum(labels_true)
        ndcg = ndcg_score(y_true.reshape((1,-1)),   # calculate ndcg score at K
                          y_pred.reshape((1,-1)),   # the rank scores must be axis 1
                          k=num_top_rank)           # each 'query' is axis 0
        
        precisions.append(f1_metrics[0])
        supports.append(sum(labels_true))
        ndcg_scores.append(ndcg)
    
    return precisions, supports, ndcg_scores


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
    
    parser.add_argument("--train_data",     default='./data/train_set_128_individual_bertweet_keywords_5.bin')
    parser.add_argument("--test_data",      default='./data/test_set_128_individual_bertweet_keywords_5.bin')
    
    parser.add_argument("--k_folds",        default=4, type=int, help='number of segments to fold training data')
    parser.add_argument("--folds2run",      default=1, type=int, help='number of times to do validation folding')
    
    parser.add_argument("--debug",          action="store_true", help="Debug flag")
    parser.add_argument("--log_interval",   default=1, type=int, help="num of batches before printing")
    ''' ===================================================='''
    ''' ========== Add additional arguments here ==========='''
    parser.add_argument('--loss_fn',        default='ce_loss',      help='loss function. ce_loss (default), dice, w_ce_loss')
    parser.add_argument('--w_sample',       action='store_true',    help='non flat sampling of training examples')
    parser.add_argument('--pretrain_model', default='',             help='model file that was pretrained on big twitter dataset')
    parser.add_argument('--epochs2giveup',  default=5, type=int,    help='training is stopped if no improvements are seen after this number of epochs')
    parser.add_argument('--dropout',        default=0.1,type=float, help='dropout probability of last layer')
    parser.add_argument('--layers',         default=2, type=int,    help='number of level 2 transformer layers')
    parser.add_argument('--viral_attr',     default='likes',        help='what attribute to use to define viral, must be ["likes", "retweets"]')
    parser.add_argument('--viral_log',      default=1, type=int,    help='Whether to log the viral counts. must be 1 or 0')
    parser.add_argument('--weight_attr',    default='stance',       help='attribute for weighted sampling. must be "stance", "viral"')
    parser.add_argument('--task',           default='multi',        help='task to train on. must be "multi", "stance", "viral"')
    parser.add_argument('--mtt_weight',     default=1.0,type=float, help='relative weight of viral task to stance task')
    parser.add_argument('--ablation',       default='',             help='features to ablate. can be "followers", "text" or "keywords"')
    ''' ===================================================='''
    return parser.parse_args()

def print_gpu_obj():
    import gc
    count = 0
    string_to_print = ''
    for tracked_obj in gc.get_objects():
        if torch.is_tensor(tracked_obj):
            if tracked_obj.is_cuda:
                count += 1
                string = str('{} {} {}'.format(type(tracked_obj).__name__, 
                                               " pinned" if tracked_obj.is_pinned() else "",
                                               tracked_obj.shape))
                string_to_print += string + '\n'
                #print('{} {} {}'.format(type(tracked_obj).__name__, 
                #                        " pinned" if tracked_obj.is_pinned() else "",
                #                        tracked_obj.shape))
    
    string_to_print += str('there are %d tracked objects in GPU' % count)
    # print('there are %d tracked objects in GPU' % count)
    return string_to_print

if __name__ == '__main__':
    main()