#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Created on Mon Mar  8 17:50:08 2021
# @author: jakeyap
# bert multi
# bert single
# bertweet single

EXP_NUM=15
for USER_WORDS in 10
do
    for layers in 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bert5 --exp_name=exp95-${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bert_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bert_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

EXP_NUM=16
for USER_WORDS in 10
do
    for layers in 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bert5 --exp_name=exp95-${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bert_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bert_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=viral --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

EXP_NUM=17
for USER_WORDS in 10
do
    for layers in 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet5 --exp_name=exp95-${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=viral --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

: "
# ablation study
EXP_NUM=1
for ABLATION in keywords followers text keywords-followers keywords-text text-followers keywords-followers-text
do
    for USER_WORDS in 10
    do
        for layers in 3
        do
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
            --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
            --model_name=mtt_Bertweet5 --exp_name=exp95-${EXP_NUM} --epochs2giveup=20 \
            --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
            --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
            --k_folds=4 --folds2run=4 \
            --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
            --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0 --ablation=${ABLATION}
        ((EXP_NUM=EXP_NUM+1))
        done
    done
done

EXP_NUM=8
for ABLATION in keywords followers text keywords-followers keywords-text text-followers keywords-followers-text
do
    for USER_WORDS in 10
    do
        for layers in 3
        do
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
            --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
            --model_name=mtt_Bertweet5 --exp_name=exp95-${EXP_NUM} --epochs2giveup=20 \
            --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
            --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
            --k_folds=4 --folds2run=4 \
            --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
            --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=viral --mtt_weight=1.0 --ablation=${ABLATION}
        ((EXP_NUM=EXP_NUM+1))
        done
    done
done
"
: "

EXP_NUM=93
for USER_WORDS in 5 10
do
    for layers in 3 2
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
        --batch_train=60 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet5 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
"
: "
EXP_NUM=85
for USER_WORDS in 5 10 15 20
do
    for layers in 3 2
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_multitask_user_features_keywords.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet4 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=90 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
"

: "
EXP_NUM=77
for USER_WORDS in 5 10 15 20
do
    for layers in 3 2
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask_user_features_keywords.py \
        --batch_train=60 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet4 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --test_data=./data/test_set_128_individual_bertweet_keywords_${USER_WORDS}.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.3 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

"

: '
EXP_NUM=65
for DROPOUT in 0.1 0.3 0.5
do
    for layers in 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet3 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=${DROPOUT} --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
'
: '
EXP_NUM=71
for DROPOUT in 0.1 0.3 0.5
do
    for layers in 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet3 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=${DROPOUT} --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
'
: "
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

EXP_NUM=53
for W_ATTR in stance viral
do
    for layers in 1 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet2 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.1 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=${W_ATTR} --task=multi --mtt_weight=1.0
    ((EXP_NUM=EXP_NUM+1))
    done
done

# to be run on screen2
EXP_NUM=59
for W_ATTR in stance viral
do
    for layers in 1 2 3
    do
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python main_multitask_user_features.py \
        --batch_train=80 --batch_test=200 --epochs=200 --learning_rate=0.00004 --optimizer=adam \
        --model_name=mtt_Bertweet2 --exp_name=exp${EXP_NUM} --epochs2giveup=20 \
        --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
        --k_folds=4 --folds2run=4 \
        --log_interval=1 --do_train --loss_fn=w_ce_loss --w_sample --dropout=0.1 --layers=${layers} \
        --viral_threshold=80 --viral_attr=likes --weight_attr=${W_ATTR} --task=multi --mtt_weight=2.0
    ((EXP_NUM=EXP_NUM+1))
    done
done
"