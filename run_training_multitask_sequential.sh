#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Created on Mon Mar  8 17:50:08 2021
# @author: jakeyap

EXP_NUM=37
for mtt_weight in 1.0 2.0
do
    for V_ATTR in likes retweets
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask_sequential.py \
        --batch_train=100 --batch_test=200 --epochs=200 --learning_rate=0.00001 --optimizer=adam \
        --model_name=mtt_Bertweet --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
        --k_folds=4 --folds2run=1 \
        --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
        --viral_threshold=80 --viral_attr=${V_ATTR} --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done