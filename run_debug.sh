#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_multitask_user_features.py \
    --batch_train=3 --batch_test=20 --epochs=2 --learning_rate=0.00001 --optimizer=adam \
    --model_name=mtt_Bertweet2 --exp_name=expXX --epochs2giveup=20 \
    --train_data=./data/train_set_128_individual_bertweet.bin --test_data=./data/test_set_128_individual_bertweet.bin \
    --k_folds=4 --folds2run=2 \
    --log_interval=1 --do_train --loss_fn=w_ce_loss --dropout=0.1 --layers=2 \
    --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0 \
    --debug
    
: "
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_multitask_sequential.py \
    --batch_train=3 --batch_test=20 --epochs=2 --learning_rate=0.00001 --optimizer=adam \
    --model_name=mtt_Bertweet --exp_name=expXX_stance_stance --epochs2giveup=20 \
    --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
    --viral_threshold=80 --viral_attr=likes --weight_attr=stance --task=multi --mtt_weight=1.0 \
    --debug
" 

: "
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_multitask.py \
    --batch_train=3 --batch_test=20 --epochs=2 --learning_rate=0.00001 --optimizer=adam \
    --model_name=mtt_Bertweet --exp_name=expXX_stance_stance --epochs2giveup=20 \
    --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
    --viral_threshold=80 --weight_attr=stance --task=stance --mtt_weight=1.0 \
    --debug
"
#MY_NUM=0
#  echo i: $i
#  ((i=i+1))
: "
#for W_ATTR in stance likes retweets
for W_ATTR in stance
do
    #for TASK in stance multi viral
    for TASK in multi
    do 
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_multitask.py \
        --batch_train=3 --batch_test=20 --epochs=2 --learning_rate=0.00001 --optimizer=adam \
        --model_name=mtt_Bertweet --exp_name=expXX_${W_ATTR}_${TASK} --epochs2giveup=20 \
        --train_data=./data/train_set_128_w_length_bertweet.bin --test_data=./data/test_set_128_w_length_bertweet.bin \
        --k_folds=4 --folds2run=1 \
        --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1 \
        --viral_threshold=80 --weight_attr=${W_ATTR} --task=${TASK} --mtt_weight=1.0 \
        --debug
    done
done
"
# weight_attr=[stance, likes, retweets]
# task=[stance, multi, viral]


: "

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=3 --batch_test=20 --epochs=3 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=expXX --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/merge_semeval_test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --pretrain_model=./log_files/saved_models/exp56_my_Bertweet.bin \
    --log_interval=1 --do_test --loss_fn=ce_loss --w_sample --dropout=0.1
    
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=3 --batch_test=20 --epochs=3 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=expXX --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/merge_srq_test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --pretrain_model=./log_files/saved_models/exp56_my_Bertweet.bin \
    --log_interval=1 --do_test --loss_fn=ce_loss --w_sample --dropout=0.1


# expXX
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=3 --batch_test=3 --epochs=3 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=expXX --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.2 --debug
"


# i=models
# k=fold
#for i in my_modelA0 
#do
#    echo ${i}
#    for k in 0 1
#    do
#        echo ${k}
#        PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#        --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
#        --output_dir ./output_release/stance_only_${i}_output10BERT_${k}_20200903/ --bert_model bert-base-uncased --do_train --do_eval \
#        --max_tweet_num 17 --max_tweet_length 30 --num_train_epochs 27
#    done
#done