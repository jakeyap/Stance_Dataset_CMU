#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

# exp21
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelE0' --exp_name='exp21' \
    --train_data='./data/train_set_256_old.bin' --test_data='./data/test_set_256_old.bin' \
    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample
    
# exp20
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelE0' --exp_name='exp20' \
    --train_data='./data/train_set_256_old.bin' --test_data='./data/test_set_256_old.bin' \
    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp19
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelE0' --exp_name='exp19' \
    --train_data='./data/train_set_256_old.bin' --test_data='./data/test_set_256_old.bin' \
    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='ce_loss'

# exp18
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelE0' --exp_name='exp18' \
    --train_data='./data/train_set_256_old.bin' --test_data='./data/test_set_256_old.bin' \
    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'


# exp17
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
#    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelE0' --exp_name='exp17' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample
    
# exp16
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
#    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelE0' --exp_name='exp16' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp15
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
#    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelE0' --exp_name='exp15' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss'

# exp14
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
#    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelE0' --exp_name='exp14' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --pretrain_model='./log_files/saved_models/pretrain02_my_modelE0.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice'
    
# exp13
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp13' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample

# exp12
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp12' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample

# exp11
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp11' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss' --w_sample

# exp10
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp10' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp09
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp09' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp08
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp08' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample
    
# exp07
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp07' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss'

# exp06
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp06' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss'

# exp05
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp05' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='ce_loss'

# exp04
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp04' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp03
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp03' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp02
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp02' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp01
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3Z python main_v2.py \
#    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
#    --model_name='my_modelA0' --exp_name='exp01' \
#    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
#    --k_folds=4 --folds2run=1 \
#    --log_interval=10 --do_train --do_test --loss_fn='dice'

