python main.py \
    --modelname dab_deformable_detr \
    --dataset_class HRSID \
    --coco_path ../datasets/HRSID/HRSID_png \
    --output_dir output/dab_deformable_detr/epoch_0_99 \
    --batch_size 2 \
    --epochs 1 \
    --start_epoch 0 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_T_max 100 \
    --lr_eta_min 1e-6 \
    --save_checkpoint_interval 30 \
    --device cuda \
    --save_log \
    --pretrain_model_path pth/DAB_Deformable_DETR/R50/checkpoint.pth \
    --finetune_ignore class_embed
    # --resume output/dab_deformable_detr/epoch_0_24/checkpoint.pth \