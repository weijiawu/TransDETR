# PRETRAIN=exps/e2e_TransVTS_r50_UnrealText/checkpoint.pth
EXP_DIR=exps/e2e_TransVTS_r50_UnrealText
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --meta_arch TransDETR_ignored \
    --dataset_file VideoText \
    --epochs 100 \
    --with_box_refine \
    --lr_drop 30 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 2 \
    --sampler_steps 1 2 3 \
    --sampler_lengths 4 4 4 5 \
    --update_query_pos \
    --rec \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path /mmu-ocr/weijiawu/Data/VideoText/MOTR\
    --data_txt_path_train ./datasets/data_path/UnrealText.train \
    --data_txt_path_val ./datasets/data_path/UnrealText.train 