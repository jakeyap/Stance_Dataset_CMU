#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Created on Mon Mar  8 17:50:08 2021
# @author: jakeyap

EXP_NUM=41
for W_ATTR in stance viral
do
    for layers in 1 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet2 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=ce_loss --w_sample --dropout=0.1 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=${W_ATTR} --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
: "
# to be run on screen2
EXP_NUM=47
for W_ATTR in stance viral
do
    for layers in 1 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet2 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=ce_loss --w_sample --dropout=0.1 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=${W_ATTR} --task=multi --mtt_weight=2.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
"