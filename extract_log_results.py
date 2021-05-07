#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:57:04 2021
For extracting summary data from log files

@author: jakeyap
"""
# TODO : make the log extractor
'''
convention inside the log file for the classification training
# line i+0 [**** Fold n results ****]
# line i+1 [---- Dev set ----]
# line i+2 [==== Predict Tweet Stance ====]
# line i+3 [Labels Prec Recall F1 Supp]
# line i+4 [Denial]
# line i+5 [Support]
# line i+6 [Comment]
# line i+7 [Queries]
# line i+8 [MacroF1]
# line i+9 [F1w_avg]
# line i+10 [Acc]
# line i+11 blank
# line i+12 [==== Predict Viral Tweets ====]
# line i+13 [Labels Prec Recall F1 Supp]
# line i+14 [N_viral]
# line i+15 [Viral]
# line i+16 [MacroF1]
# line i+17 [F1w_avg]
# line i+18 [Acc]
# line i+19 [---- Test set ----]
# line i+20 [==== Predict Tweet Stance ====]
# line i+21 [Labels Prec Recall F1 Supp]
# line i+22 [Denial]
# line i+23 [Support]
# line i+24 [Comment]
# line i+25 [Queries]
# line i+26 [MacroF1]
# line i+27 [F1w_avg]
# line i+28 [Acc]
# line i+29 blank
# line i+30 [==== Predict Viral Tweets ====]
# line i+31 [Labels Prec Recall F1 Supp]
# line i+32 [N_viral]
# line i+33 [Viral]
# line i+34 [MacroF1]
# line i+35 [F1w_avg]
# line i+36 [Acc]
'''

'''
convention inside the log file for the regression training
# line i+0 [**** Fold n results ****]
# line i+1 [---- Dev set ----]
# line i+2 [==== Predict Tweet Stance ====]
# line i+3 [Labels Prec Recall F1 Supp]
# line i+4 [Denial]
# line i+5 [Support]
# line i+6 [Comment]
# line i+7 [Queries]
# line i+8 [MacroF1]
# line i+9 [F1w_avg]
# line i+10 [Acc]
# line i+11 blank
# line i+12 [==== Predict Viral Tweets ====]
# line i+13 [TopK Prec@K Supp NDCG@K]
# line i+14 [10   0.0000   00 0.0000]
# line i+15 [20   0.0000   00 0.0000]
# line i+16 [30   0.0000   00 0.0000]
# line i+17 [40   0.0000   00 0.0000]
# line i+18 [50   0.0000   00 0.0000]
# line i+19 [MSE: 0.0000]
# line i+20 [R2E: 0.0000]
# line i+21 [---- Test set ----]
# line i+22 [==== Predict Tweet Stance ====]
# line i+23 [Labels Prec Recall F1 Supp]
# line i+24 [Denial]
# line i+25 [Support]
# line i+26 [Comment]
# line i+27 [Queries]
# line i+28 [MacroF1]
# line i+29 [F1w_avg]
# line i+30 [Acc]
# line i+31 blank
# line i+32 [==== Predict Viral Tweets ====]
# line i+33 [TopK Prec@K Supp NDCG@K]
# line i+34 [10   0.0000   00 0.0000]
# line i+35 [20   0.0000   00 0.0000]
# line i+36 [30   0.0000   00 0.0000]
# line i+37 [40   0.0000   00 0.0000]
# line i+38 [50   0.0000   00 0.0000]
# line i+39 [MSE: 0.0000]
# line i+40 [R2E: 0.0000]
'''

import re
import pandas as pd
import numpy as np

def remove_spaces(string_in):
    '''
    Removes newlines and tabs
    Parameters
    ----------
    tweet_in : string
        tweet
    Returns
    -------
    tweet_out : string
        cleaned tweet.
    '''
    re_object = re.compile('\s')
    string_out = re_object.sub(repl=' ', string=string_in)
    return string_out

def process_file(folder, fname, display=False):
    """
    Extracts summarized information from the mtt_Bertweet2 logs

    Parameters
    ----------
    folder : string
        directory name with a slash.
    fname : string
        for filename.
    display : boolean, optional
        if true, print log file summary. Defaults to false

    Returns
    -------
    df : pandas dataframe
        dataframe containing all the summarized information for each fold.

    """
    df = pd.DataFrame()
    
    with open(folder+fname, 'r') as file:
        lines = file.readlines()
        file_len = len(lines)
        re_obj = re.compile('\*\* Fold \d* results \*\*')
        
        list_dev_deny = []
        list_dev_supp = []
        list_dev_comm = []
        list_dev_query = []
        list_dev_f1_s = []
        list_dev_f1w_s = []
        list_dev_acc_s = []
        
        list_dev_nviral = []
        list_dev_viral = []
        list_dev_f1_v = []
        list_dev_f1w_v = []
        list_dev_acc_v = []
        
        list_test_deny = []
        list_test_supp = []
        list_test_comm = []
        list_test_query = []
        list_test_f1_s = []
        list_test_f1w_s = []
        list_test_acc_s = []
        
        list_test_nviral = []
        list_test_viral = []
        list_test_f1_v = []
        list_test_f1w_v = []
        list_test_acc_v = []
        
        list_of_lists = [list_dev_deny,
                         list_dev_supp,
                         list_dev_comm,
                         list_dev_query,
                         list_dev_f1_s,
                         list_dev_f1w_s,
                         list_dev_acc_s,
                         list_dev_nviral,
                         list_dev_viral,
                         list_dev_f1_v,
                         list_dev_f1w_v,
                         list_dev_acc_v,
                         list_test_deny,
                         list_test_supp,
                         list_test_comm,
                         list_test_query,
                         list_test_f1_s,
                         list_test_f1w_s,
                         list_test_acc_s,
                         list_test_nviral,
                         list_test_viral,
                         list_test_f1_v,
                         list_test_f1w_v,
                         list_test_acc_v]
        list_of_names = ['dev_deny',
                         'dev_supp',
                         'dev_comm',
                         'dev_query',
                         'dev_f1_s',
                         'dev_f1w_s',
                         'dev_acc_s',
                         'dev_nviral',
                         'dev_viral',
                         'dev_f1_v',
                         'dev_f1w_v',
                         'dev_acc_v',
                         'test_deny',
                         'test_supp',
                         'test_comm',
                         'test_query',
                         'test_f1_s',
                         'test_f1w_s',
                         'test_acc_s',
                         'test_nviral',
                         'test_viral',
                         'test_f1_v',
                         'test_f1w_v',
                         'test_acc_v']
        
        for i in range(file_len-170, file_len): # only have to look near end of log file
            line = lines[i]                     # investigate the line
            match = re_obj.search(line)         # find each validation fold's header
            if match:
                print(line, end='')
                tmp = line
                # line i+1 [---- Dev set ----]
                # line i+2 [==== Predict Tweet Stance ====]
                # line i+3 [Labels Prec Recall F1 Supp]
                dev_deny = float(lines[i+4].split('\t')[3])     # line i+4 [Denial]
                dev_supp = float(lines[i+5].split('\t')[3])     # line i+5 [Support]
                dev_comm = float(lines[i+6].split('\t')[3])     # line i+6 [Comment]
                dev_query= float(lines[i+7].split('\t')[3])     # line i+7 [Queries]
                dev_f1_s = float(lines[i+8].split('\t')[1])     # line i+8 [MacroF1]
                dev_f1w_s= float(lines[i+9].split('\t')[1])     # line i+9 [F1w_avg]
                dev_acc_s= float(lines[i+10].split('\t')[1])    # line i+10 [Acc]
                # line i+11 blank
                # line i+12 [==== Predict Viral Tweets ====]
                # line i+13 [Labels Prec Recall F1 Supp]
                dev_nviral= float(lines[i+14].split('\t')[3])   # line i+14 [N_viral]
                dev_viral = float(lines[i+15].split('\t')[3])   # line i+15 [Viral]
                dev_f1_v  = float(lines[i+16].split('\t')[1])   # line i+16 [MacroF1]
                dev_f1w_v = float(lines[i+17].split('\t')[1])   # line i+17 [F1w_avg]
                dev_acc_v = float(lines[i+18].split('\t')[1])   # line i+18 [Acc]
                # line i+19 [---- Test set ----]
                # line i+20 [==== Predict Tweet Stance ====]
                # line i+21 [Labels Prec Recall F1 Supp]
                test_deny  = float(lines[i+22].split('\t')[3])  # line i+22 [Denial]
                test_supp  = float(lines[i+23].split('\t')[3])  # line i+23 [Support]
                test_comm  = float(lines[i+24].split('\t')[3])  # line i+24 [Comment]
                test_query = float(lines[i+25].split('\t')[3])  # line i+25 [Queries]
                test_f1_s  = float(lines[i+26].split('\t')[1])  # line i+26 [MacroF1]
                test_f1w_s = float(lines[i+27].split('\t')[1])  # line i+27 [F1w_avg]
                test_acc_s = float(lines[i+28].split('\t')[1])  # line i+28 [Acc]
                # line i+29 blank
                # line i+30 [==== Predict Viral Tweets ====]
                # line i+31 [Labels Prec Recall F1 Supp]
                test_nviral= float(lines[i+32].split('\t')[3])  # line i+32 [N_viral]
                test_viral = float(lines[i+33].split('\t')[3])  # line i+33 [Viral]
                test_f1_v  = float(lines[i+34].split('\t')[1])  # line i+34 [MacroF1]
                test_f1w_v = float(lines[i+35].split('\t')[1])  # line i+35 [F1w_avg]
                test_acc_v = float(lines[i+36].split('\t')[1])  # line i+36 [Acc]
                
                if display:
                    print(dev_deny, end='\t')
                    print(dev_supp, end='\t')
                    print(dev_comm, end='\t')
                    print(dev_query, end='\t')
                    print(dev_f1_s, end='\t')
                    print(dev_f1w_s, end='\t')
                    print(dev_acc_s)
                    
                    print(dev_nviral, end='\t')
                    print(dev_viral, end='\t')
                    print(dev_f1_v, end='\t')
                    print(dev_f1w_v, end='\t')
                    print(dev_acc_v)
                    
                    print(test_deny, end='\t')
                    print(test_supp, end='\t')
                    print(test_comm, end='\t')
                    print(test_query, end='\t')
                    print(test_f1_s, end='\t')
                    print(test_f1w_s, end='\t')
                    print(test_acc_s)
                    
                    print(test_nviral, end='\t')
                    print(test_viral, end='\t')
                    print(test_f1_v, end='\t')
                    print(test_f1w_v, end='\t')
                    print(test_acc_v)
                
                list_dev_deny.append(dev_deny)
                list_dev_supp.append(dev_supp)
                list_dev_comm.append(dev_comm)
                list_dev_query.append(dev_query)
                list_dev_f1_s.append(dev_f1_s)
                list_dev_f1w_s.append(dev_f1w_s)
                list_dev_acc_s.append(dev_acc_s)
                
                list_dev_nviral.append(dev_nviral)
                list_dev_viral.append(dev_viral)
                list_dev_f1_v.append(dev_f1_v)
                list_dev_f1w_v.append(dev_f1w_v)
                list_dev_acc_v.append(dev_acc_v)
                
                list_test_deny.append(test_deny)
                list_test_supp.append(test_supp)
                list_test_comm.append(test_comm)
                list_test_query.append(test_query)
                list_test_f1_s.append(test_f1_s)
                list_test_f1w_s.append(test_f1w_s)
                list_test_acc_s.append(test_acc_s)
                
                list_test_nviral.append(test_nviral)
                list_test_viral.append(test_viral)
                list_test_f1_v.append(test_f1_v)
                list_test_f1w_v.append(test_f1w_v)
                list_test_acc_v.append(test_acc_v)
        
        # insert lists into dataframe
        for i in range(len(list_of_names)):
            name = list_of_names[i]
            data = list_of_lists[i]
            
            df.insert(loc=df.shape[1], 
                      column=name,
                      value=data)
        return df

def process_file_regression(folder, fname, display=False):
    """
    Extracts summarized information from the mtt_Bertweet2 regression logs

    Parameters
    ----------
    folder : string
        directory name with a slash.
    fname : string
        for filename.
    display : boolean, optional
        if true, print log file summary. Defaults to false

    Returns
    -------
    df : pandas dataframe
        dataframe containing all the summarized information for each fold.

    """
    df = pd.DataFrame()
    
    with open(folder+fname, 'r') as file:
        lines = file.readlines()
        file_len = len(lines)
        re_obj = re.compile('\*\* Fold \d* results \*\*')
        
        list_dev_deny = []
        list_dev_supp = []
        list_dev_comm = []
        list_dev_query = []
        list_dev_f1_s = []
        list_dev_f1w_s = []
        list_dev_acc_s = []
        list_dev_prec10 = []
        list_dev_ndcg10 = []
        list_dev_prec20 = []
        list_dev_ndcg20 = []
        list_dev_prec30 = []
        list_dev_ndcg30 = []
        list_dev_prec40 = []
        list_dev_ndcg40 = []
        list_dev_prec50 = []
        list_dev_ndcg50 = []
        list_dev_mse = []
        list_dev_r2e = []
        list_test_deny = []
        list_test_supp = []
        list_test_comm = []
        list_test_query = []
        list_test_f1_s = []
        list_test_f1w_s = []
        list_test_acc_s = []
        list_test_prec10 = []
        list_test_prec20 = []
        list_test_prec30 = []
        list_test_prec40 = []
        list_test_prec50 = []
        list_test_ndcg10 = []
        list_test_ndcg20 = []
        list_test_ndcg30 = []
        list_test_ndcg40 = []
        list_test_ndcg50 = []
        list_test_mse = []
        list_test_r2e = []
        
        list_of_lists = [list_dev_deny,
                         list_dev_supp,
                         list_dev_comm,
                         list_dev_query,
                         list_dev_f1_s,
                         list_dev_f1w_s,
                         list_dev_acc_s,
                         list_dev_prec10,
                         list_dev_ndcg10,
                         list_dev_prec20,
                         list_dev_ndcg20,
                         list_dev_prec30,
                         list_dev_ndcg30,
                         list_dev_prec40,
                         list_dev_ndcg40,
                         list_dev_prec50,
                         list_dev_ndcg50,
                         list_dev_mse,
                         list_dev_r2e,
                         list_test_deny,
                         list_test_supp,
                         list_test_comm,
                         list_test_query,
                         list_test_f1_s,
                         list_test_f1w_s,
                         list_test_acc_s,
                         list_test_prec10,
                         list_test_ndcg10,
                         list_test_prec20,
                         list_test_ndcg20,
                         list_test_prec30,
                         list_test_ndcg30,
                         list_test_prec40,
                         list_test_ndcg40,
                         list_test_prec50,
                         list_test_ndcg50,
                         list_test_mse,
                         list_test_r2e]
        list_of_names = ['dev_deny',
                         'dev_supp',
                         'dev_comm',
                         'dev_query',
                         'dev_f1_s',
                         'dev_f1w_s',
                         'dev_acc_s',
                         'dev_prec10',
                         'dev_ndcg10',
                         'dev_prec20',
                         'dev_ndcg20',
                         'dev_prec30',
                         'dev_ndcg30',
                         'dev_prec40',
                         'dev_ndcg40',
                         'dev_prec50',
                         'dev_ndcg50',
                         'dev_mse',
                         'dev_r2e',
                         'test_deny',
                         'test_supp',
                         'test_comm',
                         'test_query',
                         'test_f1_s',
                         'test_f1w_s',
                         'test_acc_s',
                         'test_prec10',
                         'test_ndcg10',
                         'test_prec20',
                         'test_ndcg20',
                         'test_prec30',
                         'test_ndcg30',
                         'test_prec40',
                         'test_ndcg40',
                         'test_prec50',
                         'test_ndcg50',
                         'test_mse',
                         'test_r2e']
        
        for i in range(file_len-170, file_len): # only have to look near end of log file
            line = lines[i]                     # investigate the line
            match = re_obj.search(line)         # find each validation fold's header
            if match:
                print(line, end='')
                tmp = line
                # line i+1 [---- Dev set ----]
                # line i+2 [==== Predict Tweet Stance ====]
                # line i+3 [Labels Prec Recall F1 Supp]
                dev_deny = float(lines[i+4].split('\t')[3])     # line i+4 [Denial]
                dev_supp = float(lines[i+5].split('\t')[3])     # line i+5 [Support]
                dev_comm = float(lines[i+6].split('\t')[3])     # line i+6 [Comment]
                dev_query= float(lines[i+7].split('\t')[3])     # line i+7 [Queries]
                dev_f1_s = float(lines[i+8].split('\t')[1])     # line i+8 [MacroF1]
                dev_f1w_s= float(lines[i+9].split('\t')[1])     # line i+9 [F1w_avg]
                dev_acc_s= float(lines[i+10].split('\t')[1])    # line i+10 [Acc]
                # line i+11 blank
                # line i+12 [==== Predict Viral Tweets ====]
                # line i+13 [TopK Prec@K Supp NDCG@K]
                dev_prec10 = float(lines[i+14].split('\t')[1])  # line i+14 [10   0.0000   00 0.0000]
                dev_ndcg10 = float(lines[i+14].split('\t')[3])  
                dev_prec20 = float(lines[i+15].split('\t')[1])  # line i+15 [20   0.0000   00 0.0000]
                dev_ndcg20 = float(lines[i+15].split('\t')[3])  
                dev_prec30 = float(lines[i+16].split('\t')[1])  # line i+16 [30   0.0000   00 0.0000]
                dev_ndcg30 = float(lines[i+16].split('\t')[3])  
                dev_prec40 = float(lines[i+17].split('\t')[1])  # line i+17 [40   0.0000   00 0.0000]
                dev_ndcg40 = float(lines[i+17].split('\t')[3])  
                dev_prec50 = float(lines[i+18].split('\t')[1])  # line i+18 [50   0.0000   00 0.0000]
                dev_ndcg50 = float(lines[i+18].split('\t')[3])  
                dev_mse = float(lines[i+19].split('\t')[1])     # line i+19 [MSE: 0.0000]
                dev_r2e = float(lines[i+20].split('\t')[1])     # line i+20 [R2E: 0.0000]
                # line i+21 [---- Test set ----]
                # line i+22 [==== Predict Tweet Stance ====]
                # line i+23 [Labels Prec Recall F1 Supp]
                test_deny  = float(lines[i+24].split('\t')[3])  # line i+24 [Denial]
                test_supp  = float(lines[i+25].split('\t')[3])  # line i+25 [Support]
                test_comm  = float(lines[i+26].split('\t')[3])  # line i+26 [Comment]
                test_query = float(lines[i+27].split('\t')[3])  # line i+27 [Queries]
                test_f1_s  = float(lines[i+28].split('\t')[1])  # line i+28 [MacroF1]
                test_f1w_s = float(lines[i+29].split('\t')[1])  # line i+29 [F1w_avg]
                test_acc_s = float(lines[i+30].split('\t')[1])  # line i+30 [Acc]
                # line i+31 blank
                # line i+32 [==== Predict Viral Tweets ====]
                # line i+33 [TopK Prec@K Supp NDCG@K]
                test_prec10 = float(lines[i+34].split('\t')[1]) # line i+34 [10   0.0000   00 0.0000]
                test_ndcg10 = float(lines[i+34].split('\t')[3])  
                test_prec20 = float(lines[i+35].split('\t')[1]) # line i+35 [20   0.0000   00 0.0000]
                test_ndcg20 = float(lines[i+35].split('\t')[3])  
                test_prec30 = float(lines[i+36].split('\t')[1]) # line i+36 [30   0.0000   00 0.0000]
                test_ndcg30 = float(lines[i+36].split('\t')[3])  
                test_prec40 = float(lines[i+37].split('\t')[1]) # line i+37 [40   0.0000   00 0.0000]
                test_ndcg40 = float(lines[i+37].split('\t')[3])  
                test_prec50 = float(lines[i+38].split('\t')[1]) # line i+38 [50   0.0000   00 0.0000]
                test_ndcg50 = float(lines[i+38].split('\t')[3])  
                test_mse = float(lines[i+39].split('\t')[1])    # line i+39 [MSE: 0.0000]
                test_r2e = float(lines[i+40].split('\t')[1])    # line i+40 [R2E: 0.0000]
                # TODO : reached here
                
                if display:
                    print(dev_deny, end='\t')
                    print(dev_supp, end='\t')
                    print(dev_comm, end='\t')
                    print(dev_query, end='\t')
                    print(dev_f1_s, end='\t')
                    print(dev_f1w_s, end='\t')
                    print(dev_acc_s)
                    
                    print("%1.4f\t%1.4f" % (dev_prec10, dev_ndcg10))
                    print("%1.4f\t%1.4f" % (dev_prec20, dev_ndcg20))
                    print("%1.4f\t%1.4f" % (dev_prec30, dev_ndcg30))
                    print("%1.4f\t%1.4f" % (dev_prec40, dev_ndcg40))
                    print("%1.4f\t%1.4f" % (dev_prec50, dev_ndcg50))
                    print("%1.4f" % (dev_mse))
                    print("%1.4f" % (dev_r2e))
                    
                    print(test_deny, end='\t')
                    print(test_supp, end='\t')
                    print(test_comm, end='\t')
                    print(test_query, end='\t')
                    print(test_f1_s, end='\t')
                    print(test_f1w_s, end='\t')
                    print(test_acc_s)
                    
                    print("%1.4f\t%1.4f" % (test_prec10, test_ndcg10))
                    print("%1.4f\t%1.4f" % (test_prec20, test_ndcg20))
                    print("%1.4f\t%1.4f" % (test_prec30, test_ndcg30))
                    print("%1.4f\t%1.4f" % (test_prec40, test_ndcg40))
                    print("%1.4f\t%1.4f" % (test_prec50, test_ndcg50))
                    print("%1.4f" % (test_mse))
                    print("%1.4f" % (test_r2e))
                
                list_dev_deny.append(dev_deny)
                list_dev_supp.append(dev_supp)
                list_dev_comm.append(dev_comm)
                list_dev_query.append(dev_query)
                list_dev_f1_s.append(dev_f1_s)
                list_dev_f1w_s.append(dev_f1w_s)
                list_dev_acc_s.append(dev_acc_s)
                
                list_dev_prec10.append(dev_prec10)
                list_dev_ndcg10.append(dev_ndcg10)
                list_dev_prec20.append(dev_prec20)
                list_dev_ndcg20.append(dev_ndcg20)
                list_dev_prec30.append(dev_prec30)
                list_dev_ndcg30.append(dev_ndcg30)
                list_dev_prec40.append(dev_prec40)
                list_dev_ndcg40.append(dev_ndcg40)
                list_dev_prec50.append(dev_prec50)
                list_dev_ndcg50.append(dev_ndcg50)
                list_dev_mse.append(dev_mse)
                list_dev_r2e.append(dev_r2e)
                
                list_test_deny.append(test_deny)
                list_test_supp.append(test_supp)
                list_test_comm.append(test_comm)
                list_test_query.append(test_query)
                list_test_f1_s.append(test_f1_s)
                list_test_f1w_s.append(test_f1w_s)
                list_test_acc_s.append(test_acc_s)
                
                list_test_prec10.append(test_prec10)
                list_test_ndcg10.append(test_ndcg10)
                list_test_prec20.append(test_prec20)
                list_test_ndcg20.append(test_ndcg20)
                list_test_prec30.append(test_prec30)
                list_test_ndcg30.append(test_ndcg30)
                list_test_prec40.append(test_prec40)
                list_test_ndcg40.append(test_ndcg40)
                list_test_prec50.append(test_prec50)
                list_test_ndcg50.append(test_ndcg50)
                list_test_mse.append(test_mse)
                list_test_r2e.append(test_r2e)
        
        # insert lists into dataframe
        for i in range(len(list_of_names)):
            name = list_of_names[i]
            data = list_of_lists[i]
            
            df.insert(loc=df.shape[1], 
                      column=name,
                      value=data)
        return df
# line i+0 [**** Fold n results ****]
# line i+1 [---- Dev set ----]
# line i+2 [==== Predict Tweet Stance ====]
# line i+3 [Labels Prec Recall F1 Supp]
# line i+4 [Denial]
# line i+5 [Support]
# line i+6 [Comment]
# line i+7 [Queries]
# line i+8 [MacroF1]
# line i+9 [F1w_avg]
# line i+10 [Acc]
# line i+11 blank
# line i+12 [==== Predict Viral Tweets ====]
# line i+13 [TopK Prec@K Supp NDCG@K]
# line i+14 [10   0.0000   00 0.0000]
# line i+15 [20   0.0000   00 0.0000]
# line i+16 [30   0.0000   00 0.0000]
# line i+17 [40   0.0000   00 0.0000]
# line i+18 [50   0.0000   00 0.0000]
# line i+19 [MSE: 0.0000]
# line i+20 [R2E: 0.0000]
# line i+21 [---- Test set ----]
# line i+22 [==== Predict Tweet Stance ====]
# line i+23 [Labels Prec Recall F1 Supp]
# line i+24 [Denial]
# line i+25 [Support]
# line i+26 [Comment]
# line i+27 [Queries]
# line i+28 [MacroF1]
# line i+29 [F1w_avg]
# line i+30 [Acc]
# line i+31 blank
# line i+32 [==== Predict Viral Tweets ====]
# line i+33 [TopK Prec@K Supp NDCG@K]
# line i+34 [10   0.0000   00 0.0000]
# line i+35 [20   0.0000   00 0.0000]
# line i+36 [30   0.0000   00 0.0000]
# line i+37 [40   0.0000   00 0.0000]
# line i+38 [50   0.0000   00 0.0000]
# line i+39 [MSE: 0.0000]
# line i+40 [R2E: 0.0000]

def print_averages(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    averages = []
    col_width = len(columns)
    
    for i in range(col_width):
        col_title = columns[i]
        col_value = np.average(df[col_title])
        averages.append(col_value)
        print(col_title, end='\t') # print titles in a row
    
    print()     # print on next line
    for i in range(col_width):
        print(('%1.4f' % averages[i]), end='\t')
    print()     # print on next line
    return

'''
teststr = '******************** Fold 100 results ********************\n'
re_obj = re.compile('Fold \d* results')
#re_obj = re.compile('Fold')
str_out = re_obj.search(teststr)
'''
if __name__=='__main__':
    folder = './log_files/'
    fnames = ['exp32_mtt_Bertweet5_regr.log',
              'exp33_mtt_Bertweet5_regr.log',
              'exp34_mtt_Bertweet5_regr.log']
    for fname in fnames:
        tmp = process_file_regression(folder, fname)
        print_averages(tmp)