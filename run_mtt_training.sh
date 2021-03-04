#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Created on Thu Mar  4 18:34:11 2021

#@author: jakeyap


# exp01-exp09
EXP_NUM=1
for W_ATTR in stance likes retweets
do
    for TASK in multi viral stance
    do 
    echo EXP_NUM: ${EXP_NUM} 
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask.py \
        --batch_train=100 --batch_test=300 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet --exp_name=exp0${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
        --k_folds=4 --folds2run=1 \
        --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
        --viral_threshold=80 --weight_attr=${W_ATTR} --task=${TASK} --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

# exp10-exp27
for THRESHOLD in 70 80 90
do
    for W_ATTR in likes retweets
    do
        for MTT_WEIGHT in 0.5 1.0 2.0
        do 
        echo EXP_NUM: ${EXP_NUM} 
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask.py \
            --batch_train=100 --batch_test=300 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
            --model_name=mtt_Bertweet --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
            --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
            --k_folds=4 --folds2run=1 \
            --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
            --viral_threshold=${THRESHOLD} --weight_attr=${W_ATTR} --task=multi --mtt_weight=${MTT_WEIGHT}
        ((EXP_NUM=EXP_NUM+1))
        done
    done
done
