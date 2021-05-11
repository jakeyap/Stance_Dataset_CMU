#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Created on Thu Apr 29 20:21:44 2021
# @author: jakeyap
# bert multi
# bert single
# bertweet single

# for quick verification of hyperparams
# must use log. normal mode sucks, cant train at all

# ablation study
EXP_NUM=1
for ABLATION in keywords followers text keywords-followers keywords-text text-followers keywords-followers-text
do
    for LOGVIRAL in 1
    do
        for DROPOUT in 0.3
        do
            for USER_WORDS in 10
            do
                for LAYERS in 3
                do
                PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_regression.py \
                    --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
                    --model_name=mtt_Bertweet5_regr --exp_name=exp7-${EXP_NUM} --epochs2giveup=10 \
                    --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
                    --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
                    --k_folds=4 --folds2run=4 \
                    --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=${DROPOUT} --layers=${LAYERS} \
                    --viral_log=${LOGVIRAL} --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=0.31 --ablation=${ABLATION}
                ((EXP_NUM=EXP_NUM+1))
                done
            done
        done
    done
done

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_regression.py \
                    --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
                    --model_name=mtt_Bertweet5_regr --exp_name=exp7-8 --epochs2giveup=10 \
                    --train_data=./data/train_set_128_individual_bertweet_keywords_10.bin \
                    --test_data=./data/test_set_128_individual_bertweet_keywords_10.bin \
                    --k_folds=4 --folds2run=4 \
                    --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=3 \
                    --viral_log=1 --viral_attr=likes --weight_attr=stance --task=viral --mtt_weight=0.31

: "
EXP_NUM=17
for LOGVIRAL in 1
do
    for DROPOUT in 0.5 0.7
    do
        for USER_WORDS in 10 15
        do
            for LAYERS in 3
            do
                for MTT_WT in 0.10 0.17 0.31 0.5 1 1.7 3.1
                do
                PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_regression.py \
                    --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
                    --model_name=mtt_Bertweet5_regr --exp_name=exp${EXP_NUM} --epochs2giveup=10 \
                    --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
                    --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
                    --k_folds=4 --folds2run=4 \
                    --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=${DROPOUT} --layers=${LAYERS} \
                    --viral_log=${LOGVIRAL} --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=${MTT_WT}
                ((EXP_NUM=EXP_NUM+1))
                done
            done
        done
    done
done
"