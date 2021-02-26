#!/usr/bin/env bash
#@author: jakeyap on 20210208 1100am


# ===================== the semevel dataset =====================
# exp56
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=50 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp56 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# exp55
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=50 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp55 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# exp54
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=50 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp54 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# exp53
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=100 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp53 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# exp52
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=100 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp52 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# exp51
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=100 --batch_test=300 --epochs=100 --learning_rate=0.000020 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp51 --epochs2giveup=20 \
    --train_data=./data/train_set_128_semeval17_bertweet.bin --test_data=./data/test_set_128_semeval17_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

: '
# ===============================================================

# exp50
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00010 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp50 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.3

# exp49
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00010 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp49 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.25
    
# exp48
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00010 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp48 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.2
    
# exp47
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00010 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp47 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.15

# exp46
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00010 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp46 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample --dropout=0.1

# =====================================================================


# exp45
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=100 --batch_test=100 --epochs=100 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp45 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp44
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=100 --batch_test=100 --epochs=100 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp44 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp43
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=100 --batch_test=100 --epochs=100 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp43 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp42
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=100 --batch_test=100 --epochs=100 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp42 --epochs2giveup=20\
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp41
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=200 --batch_test=200 --epochs=100 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp41 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp40
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=200 --batch_test=200 --epochs=100 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp40 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp39
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=200 --batch_test=200 --epochs=100 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp39 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp38
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=200 --batch_test=200 --epochs=100 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp38 --epochs2giveup=20\
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp37
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp37 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp36
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp36 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp35
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp35 --epochs2giveup=20 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp34
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=100 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp34 --epochs2giveup=20\
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp33
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp33 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp32
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp32 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=dice --w_sample

# exp31
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp31 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss

# exp30
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp30 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=dice
    
# exp29
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp29 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp28
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp28 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=dice --w_sample

# exp27
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp27 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=ce_loss

# exp26
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=300 --batch_test=300 --epochs=40 --learning_rate=0.000005 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp26 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=1 --do_train --do_test --loss_fn=dice

# exp25
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp25 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp24
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp24 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample

# exp23
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp23 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp22
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,0 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_Bertweet --exp_name=exp22 \
    --train_data=./data/train_set_128_bertweet.bin --test_data=./data/test_set_128_bertweet.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice
# exp21
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp21 \
    --train_data=./data/train_set_256_old.bin --test_data=./data/test_set_256_old.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp20
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp20 \
    --train_data=./data/train_set_256_old.bin --test_data=./data/test_set_256_old.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample

# exp19
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp19 \
    --train_data=./data/train_set_256_old.bin --test_data=./data/test_set_256_old.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp18
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp18 \
    --train_data=./data/train_set_256_old.bin --test_data=./data/test_set_256_old.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice


# exp17
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp17 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample
    
# exp16
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp16 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample

# exp15
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp15 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp14
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1,3 python main_v2.py \
    --batch_train=128 --batch_test=128 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelE0 --exp_name=exp14 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --pretrain_model=./log_files/saved_models/pretrain02_my_modelE0.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice
   
# exp13
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp13 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp12
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp12 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp11
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp11 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss --w_sample

# exp10
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp10 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample

# exp09
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp09 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample

# exp08
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp08 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice --w_sample
    
# exp07
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp07 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp06
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp06 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp05
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp05 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=ce_loss

# exp04
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00008 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp04 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice

# exp03
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00004 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp03 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice

# exp02
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00002 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp02 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice

# exp01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3Z python main_v2.py \
    --batch_train=64 --batch_test=64 --epochs=40 --learning_rate=0.00001 --optimizer=adam \
    --model_name=my_modelA0 --exp_name=exp01 \
    --train_data=./data/train_set_256_new.bin --test_data=./data/test_set_256_new.bin \
    --k_folds=4 --folds2run=1 \
    --log_interval=10 --do_train --do_test --loss_fn=dice

'