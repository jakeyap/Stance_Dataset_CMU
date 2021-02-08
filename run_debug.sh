#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

# exp01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=2 --batch_test=40 --epochs=20 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='expXX' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --debug --loss_fn='dice' --w_sample

