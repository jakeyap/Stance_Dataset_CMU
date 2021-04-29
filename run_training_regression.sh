#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Created on Thu Apr 29 20:21:44 2021
# @author: jakeyap
# bert multi
# bert single
# bertweet single

EXP_NUM=1
for USER_WORDS in 10
do
    for layers in 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_regression.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet5_regr --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bert_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_log=1 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done