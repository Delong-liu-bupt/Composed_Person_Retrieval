#!/bin/bash
# FAFA Model Inference Script
# Example usage for running inference on a trained model

# Run inference on the trained model
python inference_fafa.py \
    --itcpr-root /your/custom/itcpr/root \
    --exp-dir /mnt/cache/liudelong/codes/FAFA2/output/cpr/2025-09-22_17:48:02_FAFA_SynCPR_FDA_FD_MFR \
    --model-name tuned_recall_at1_step.pt \
    --dataset itcpr \
    --batch-size 256 \
    --num-workers 4 \
    --device cuda