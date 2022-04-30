# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang Univeristy-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

EXP_DIR=exps/e2e_TransVTS_r50_BOVText
# EXP_DIR=exps/e2e_TransVTS_r50_SynthText
# EXP_DIR=exps/e2e_TransVTS_r50_COCOTextV2
python3 eval.py \
    --meta_arch TransDETR_ignored \
    --dataset_file VideoText \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 3 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path /share/wuweijia/Data/VideoText/MOTR\
    --data_txt_path_train ./datasets/data_path/BOVText.train \
    --data_txt_path_val ./datasets/data_path/BOVText.train \
    --resume ${EXP_DIR}/checkpoint.pth \
    --show
    


# YVT
# EXP_DIR=exps/e2e_TransVTS_r50_YVT
# python3 eval.py \
#     --meta_arch TransVTS \
#     --dataset_file Text \
#     --epoch 200 \
#     --with_box_refine \
#     --lr_drop 100 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${EXP_DIR}/motr_final.pth \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 5 \
#     --sampler_steps 50 90 120 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --query_interaction_layer 'QIM' \
#     --extra_track_attn \
#     --mot_path /share/wuweijia/Data/VideoText/MOTR\
#     --data_txt_path_train ./datasets/data_path/YVT.train \
#     --data_txt_path_val ./datasets/data_path/YVT.train \
#     --resume ${EXP_DIR}/checkpoint0007.pth
#     --show

# minetto
# EXP_DIR=exps/e2e_TransVTS_r50_minetto
# python3 eval.py \
#     --meta_arch TransVTS \
#     --dataset_file Text \
#     --epoch 200 \
#     --with_box_refine \
#     --lr_drop 100 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${EXP_DIR}/motr_final.pth \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 5 \
#     --sampler_steps 50 90 120 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --query_interaction_layer 'QIM' \
#     --extra_track_attn \
#     --mot_path /share/wuweijia/Data/VideoText/MOTR\
#     --data_txt_path_train ./datasets/data_path/minetto.train \
#     --data_txt_path_val ./datasets/data_path/minetto.train \
#     --resume exps/e2e_TransVTS_r50_ICDAR15/checkpoint0040.pth
#     --show