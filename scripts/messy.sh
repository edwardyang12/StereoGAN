#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path="/cephfs/jianyu/eval/cs_eval"

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi


python -m torch.distributed.launch --nproc_per_node=1 /cephfs/jianyu/cs_train/CasStereoNet/main.py \
    --dataset messy_table \
    --test_dataset messy_table \
    --datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --trainlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_train.txt \
    --test_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_val.txt \
    --epochs 300 \
    --lrepochs "200:10" \
    --crop_width 512  \
    --crop_height 256 \
    --test_crop_width 1248  \
    --test_crop_height 768 \
    --ndisp "48,24" \
    --disp_inter_r "4,1" \
    --dlossw "0.5,2.0"  \
    --using_ns \
    --ns_size 3 \
    --model gwcnet-c \
    --logdir "/cephfs/jianyu/eval/cs_eval_gan"  \
    --ndisps "48,24" \
    --disp_inter_r "4,1"  \
    --batch_size 2 \
    --mode train \
    --summary_freq 50 \
    --test_summary_freq 500 \
    --brightness 0.5 \
    --contrast 0.5 \
    --kernel 3 \
    --var "0.1,2.0" \
    #--loadckpt "/cephfs/jianyu/train/cs_train/checkpoint_best.ckpt"
