#!/bin/bash

python main.py \
    --modelname dab_deformable_detr \
    --dilation \
    --dataset_class HRSID \
    --coco_path ../autodl-tmp/HRSID/HRSID_png \
    --output_dir output/dab_deformable_detr/fine_tuning_modifyparams/epoch_0_49 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_T_max 100 \
    --lr_eta_min 1e-6 \
    --focal_alpha 0.2 \
    --bbox_loss_coef 6 \
    --giou_loss_coef 3 \
    --save_checkpoint_interval 20 \
    --device cuda \
    --save_log \
    --pretrain_model_path ../model_zoo/DAB_Deformable_DETR/R50/checkpoint.pth \
    --finetune_ignore class_embed
    # --resume output/dab_deformable_detr/epoch_0_24/checkpoint.pth \