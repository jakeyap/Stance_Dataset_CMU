#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

# exp01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=5 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='expXX' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --debug --loss_fn='dice' --w_sample

# i=models
# k=fold
for i in 'my_modelA0' 
do
    echo ${i}
    for k in '0'
    do
        echo ${k}
        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
        --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
        --output_dir ./output_release/stance_only_${i}_output10BERT_${k}_20200903/ --bert_model bert-base-uncased --do_train --do_eval \
        --max_tweet_num 17 --max_tweet_length 30 --num_train_epochs 27
    done
done