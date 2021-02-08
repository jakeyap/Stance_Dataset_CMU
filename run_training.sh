#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am

# exp16
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp16' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp15
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp15' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp14
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp14' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp13
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp13' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp12
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp12' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp11
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp011' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp10
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp010' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp09
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp09' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp08
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp08' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice' --w_sample

# exp07
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp07' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp06
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp06' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp05
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp05' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp04
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp04' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp03
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp03' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp02
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp02' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

# exp01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer='adam' \
    --model_name='my_modelA0' --exp_name='exp01' \
    --train_data='./data/train_set_256_new.bin' --test_data='./data/test_set_256_new.bin' \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn='dice'

