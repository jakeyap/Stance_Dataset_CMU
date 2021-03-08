#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Created on Thu Mar  4 18:34:11 2021

#@author: jakeyap


: '
# exp1-exp12
EXP_NUM=1
for W_ATTR in stance viral 
do
    for TASK in multi viral stance
    do 
        for V_ATTR in likes retweets
        do
        echo EXP_NUM: ${EXP_NUM} 
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python main_multitask.py \
            --batch_train=100 --batch_test=300 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
            --model_name=mtt_Bertweet --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
            --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
            --k_folds=4 --folds2run=1 \
            --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
            --viral_attr=${V_ATTR} --viral_threshold=80 --weight_attr=${W_ATTR} --task=${TASK} --mtt_weight=1.0
        ((EXP_NUM=EXP_NUM+1))
        done
    done
done
'

EXP_NUM=13
# exp13-exp36
for THRESHOLD in 70 80 90
do
    for W_ATTR in stance viral
    do
        for MTT_WEIGHT in 1.0 2.0
        do 
            for V_ATTR in likes retweets
            do
            echo EXP_NUM: ${EXP_NUM} 
            PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask.py \
                --batch_train=100 --batch_test=300 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
                --model_name=mtt_Bertweet --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
                --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
                --k_folds=4 --folds2run=1 \
                --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
                --viral_attr=${V_ATTR} --viral_threshold=${THRESHOLD} --weight_attr=${W_ATTR} --task=multi --mtt_weight=${MTT_WEIGHT}
            ((EXP_NUM=EXP_NUM+1))
            done
        done
    done
done
