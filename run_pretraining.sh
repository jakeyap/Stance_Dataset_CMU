#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

# pretrain01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python pretrain_model.py \
    --batch_train=128 --batch_test=256 --epochs=1 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelE0' --exp_name='pretrain02' \
    --train_data='./../Data/SRQ_Stance_Twitter/event_universe_valid_encoded_256.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample --debug

