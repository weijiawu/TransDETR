# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang Univeristy-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


EXP_DIR=exps/e2e_TransVTS_r50_minetto
python3 eval.py \
    --meta_arch TransDETR_ignored \
    --dataset_file Text \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${EXP_DIR}/motr_final.pth \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 5 \
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
    --data_txt_path_train ./datasets/data_path/minetto.train \
    --data_txt_path_val ./datasets/data_path/minetto.train \
    --resume exps/e2e_TransVTS_r50_FlowText/checkpointMOTA27.4_IDF47.9.pth
#     --resume exps/e2e_TransVTS_r50_ICDAR15/checkpoint_FlowText_MOTA48.5IDF163.1.pth
    
    #--resume exps/e2e_TransVTS_r50_VISD/checkpointMOTA0.214IDF144.7.pth
    #--resume exps/e2e_TransVTS_r50_FlowTextV2/checkpoint0018.pth
