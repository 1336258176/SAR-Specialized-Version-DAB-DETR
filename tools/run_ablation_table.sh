#!/usr/bin/env bash
set -euo pipefail

# Reproducible low-budget ablation runner for undergraduate thesis experiments.
# It does NOT modify the original train.sh settings.
#
# Usage:
#   bash tools/run_ablation_table.sh all
#   bash tools/run_ablation_table.sh baseline
#   bash tools/run_ablation_table.sh sar_data
#   bash tools/run_ablation_table.sh full
#
# Notes:
# - Default uses `python` to keep consistency with train.sh.
# - You can override PYTHON_BIN if your environment only has python3:
#     PYTHON_BIN=python3 bash tools/run_ablation_table.sh all

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_FILE="main.py"

# ===== Base args: copied from train.sh, unchanged =====
BASE_ARGS=(
  --modelname dab_deformable_detr
  --dilation
  --dataset_class HRSID
  --coco_path ../autodl-tmp/HRSID/HRSID_png
  --batch_size 4
  --epochs 50
  --lr 1e-4
  --lr_backbone 1e-5
  --lr_T_max 100
  --lr_eta_min 1e-6
  --focal_alpha 0.2
  --bbox_loss_coef 6
  --giou_loss_coef 3
  --save_checkpoint_interval 20
  --device cuda
  --save_log
  --pretrain_model_path ../model_zoo/DAB_Deformable_DETR/R50/checkpoint.pth
  --finetune_ignore class_embed
)

ROOT_OUT="output/dab_deformable_detr/ablation_table_2026_lite"

run_train() {
  local exp_id="$1"
  local exp_name="$2"
  shift 2
  local extra_args=("$@")

  local out_dir="${ROOT_OUT}/${exp_id}_${exp_name}"
  echo "[Ablation][Train] ${exp_id} ${exp_name}"
  echo "Output: ${out_dir}"

  "${PYTHON_BIN}" "${MAIN_FILE}" \
    "${BASE_ARGS[@]}" \
    --output_dir "${out_dir}" \
    "${extra_args[@]}"
}

run_eval_tta() {
  local exp_dir_name="$1"
  local ckpt="${ROOT_OUT}/${exp_dir_name}/checkpoint.pth"
  local out_dir="${ROOT_OUT}/${exp_dir_name}/eval_tta"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[Ablation][Eval-TTA] checkpoint not found: ${ckpt}"
    exit 1
  fi

  echo "[Ablation][Eval-TTA] ${exp_dir_name}"
  echo "Checkpoint: ${ckpt}"
  echo "Output: ${out_dir}"

  "${PYTHON_BIN}" "${MAIN_FILE}" \
    "${BASE_ARGS[@]}" \
    --eval \
    --resume "${ckpt}" \
    --output_dir "${out_dir}" \
    --enable_tta_eval \
    --tta_scales 1.0 1.15 1.3 \
    --tta_nms_iou_thresh 0.6 \
    --tta_score_thresh 0.05 \
    --tta_topk 100
}

show_plan() {
  cat << 'EOF'
Low-budget 5-group ablation set:
  exp00_baseline    : Baseline (no new module)
  exp01_sar_data    : Data-level SAR augmentation bundle (E)
  exp02_model_stab  : Model/training stabilization bundle (A + C)
  exp03_loss_sar    : Loss-level SAR geometry bundle (B + F)
  exp04_full        : Recommended full lightweight combo (E + A + C + B + F)

Optional (not counted in 5 groups):
  eval_tta <exp_dir_name>
  Example: bash tools/run_ablation_table.sh eval_tta exp04_full
EOF
}

cmd="${1:-all}"
case "${cmd}" in
  plan)
    show_plan
    ;;

  baseline|exp00_baseline)
    run_train exp00 baseline
    ;;

  sar_data|exp01_sar_data)
    run_train exp01 sar_data \
      --enable_sar_vertical_flip \
      --sar_vertical_flip_prob 0.5 \
      --enable_sar_speckle_aug \
      --sar_speckle_prob 0.35 \
      --sar_speckle_min_std 0.02 \
      --sar_speckle_max_std 0.10 \
      --enable_sar_contrast_stretch \
      --sar_contrast_stretch_prob 0.35 \
      --sar_contrast_stretch_lower_q 0.02 \
      --sar_contrast_stretch_upper_q 0.98
    ;;

  model_stab|exp02_model_stab)
    run_train exp02 model_stab \
      --enable_ema \
      --ema_decay 0.9997 \
      --use_ema_for_eval \
      --enable_backbone_warmup \
      --backbone_warmup_epochs 5
    ;;

  loss_sar|exp03_loss_sar)
    run_train exp03 loss_sar \
      --enable_small_object_reweight \
      --small_object_reweight_alpha 1.0 \
      --small_object_reweight_power 0.5 \
      --enable_sar_shape_prior_loss \
      --sar_shape_prior_loss_coef 0.3
    ;;

  full|exp04_full)
    run_train exp04 full \
      --enable_sar_vertical_flip \
      --sar_vertical_flip_prob 0.5 \
      --enable_sar_speckle_aug \
      --sar_speckle_prob 0.35 \
      --sar_speckle_min_std 0.02 \
      --sar_speckle_max_std 0.10 \
      --enable_sar_contrast_stretch \
      --sar_contrast_stretch_prob 0.35 \
      --sar_contrast_stretch_lower_q 0.02 \
      --sar_contrast_stretch_upper_q 0.98 \
      --enable_ema \
      --ema_decay 0.9997 \
      --use_ema_for_eval \
      --enable_backbone_warmup \
      --backbone_warmup_epochs 5 \
      --enable_small_object_reweight \
      --small_object_reweight_alpha 1.0 \
      --small_object_reweight_power 0.5 \
      --enable_sar_shape_prior_loss \
      --sar_shape_prior_loss_coef 0.3
    ;;

  eval_tta)
    if [[ $# -lt 2 ]]; then
      echo "Usage: bash tools/run_ablation_table.sh eval_tta <exp_dir_name>"
      echo "Example: bash tools/run_ablation_table.sh eval_tta exp04_full"
      exit 1
    fi
    run_eval_tta "$2"
    ;;

  all)
    run_train exp00 baseline
    run_train exp01 sar_data \
      --enable_sar_vertical_flip \
      --sar_vertical_flip_prob 0.5 \
      --enable_sar_speckle_aug \
      --sar_speckle_prob 0.35 \
      --sar_speckle_min_std 0.02 \
      --sar_speckle_max_std 0.10 \
      --enable_sar_contrast_stretch \
      --sar_contrast_stretch_prob 0.35 \
      --sar_contrast_stretch_lower_q 0.02 \
      --sar_contrast_stretch_upper_q 0.98
    run_train exp02 model_stab \
      --enable_ema \
      --ema_decay 0.9997 \
      --use_ema_for_eval \
      --enable_backbone_warmup \
      --backbone_warmup_epochs 5
    run_train exp03 loss_sar \
      --enable_small_object_reweight \
      --small_object_reweight_alpha 1.0 \
      --small_object_reweight_power 0.5 \
      --enable_sar_shape_prior_loss \
      --sar_shape_prior_loss_coef 0.3
    run_train exp04 full \
      --enable_sar_vertical_flip \
      --sar_vertical_flip_prob 0.5 \
      --enable_sar_speckle_aug \
      --sar_speckle_prob 0.35 \
      --sar_speckle_min_std 0.02 \
      --sar_speckle_max_std 0.10 \
      --enable_sar_contrast_stretch \
      --sar_contrast_stretch_prob 0.35 \
      --sar_contrast_stretch_lower_q 0.02 \
      --sar_contrast_stretch_upper_q 0.98 \
      --enable_ema \
      --ema_decay 0.9997 \
      --use_ema_for_eval \
      --enable_backbone_warmup \
      --backbone_warmup_epochs 5 \
      --enable_small_object_reweight \
      --small_object_reweight_alpha 1.0 \
      --small_object_reweight_power 0.5 \
      --enable_sar_shape_prior_loss \
      --sar_shape_prior_loss_coef 0.3
    ;;

  *)
    echo "Unknown command: ${cmd}"
    show_plan
    exit 1
    ;;
esac
