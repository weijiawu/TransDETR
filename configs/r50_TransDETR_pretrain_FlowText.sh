# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.   sample_interval 5
# ------------------------------------------------------------------------

# training ICDAR2015  e2e_TransVTS_r50_SynthText  e2e_TransVTS_r50_COCOTextV2
# EXP_DIR=exps/e2e_TransVTS_r50_VideoSynthText
# EXP_DIR=exps/e2e_TransVTS_r50_FlowText
EXP_DIR=exps/e2e_TransVTS_r50_FlowTextV2
# PRETRAIN=exps/e2e_TransVTS_r50_FlowText/checkpoint.pth

python3 -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --meta_arch TransDETR_ignored \
    --dataset_file VideoText \
    --epochs 10 \
    --with_box_refine \
    --lr_drop 5 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 5 \
    --sampler_steps 1 2 \
    --sampler_lengths 7 8 9 \
    --update_query_pos \
    --rec\
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path /mmu-ocr/weijiawu/Data/VideoText/MOTR\
    --data_txt_path_train ./datasets/data_path/FlowTextV2.train \
    --data_txt_path_val ./datasets/data_path/FlowTextV2.train 
