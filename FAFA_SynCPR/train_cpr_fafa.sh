# FAFA Training on SynCPR Dataset for Composed Person Retrieval
CUDA_VISIBLE_DEVICES=0 \
python src/blip_fine_tune_new.py \
   --dataset cpr \
   --syncpr-data-path /your/custom/syncpr/root \
   --itcpr-root /your/custom/itcpr/root \
   --json-path SynCPR.json \
   --exp-name FAFA_SynCPR_FDA_FD_MFR \
   --blip-model-name blip2_fafa_cpr \
   --setting annotations \
   --num-epochs 10 \
   --num-workers 4 \
   --learning-rate 2e-6 \
   --batch-size 256 \
   --transform squarepad \
   --save-training \
   --save-best \
   --validation-frequency 1 \
   --validation-step 500 \
   --loss-fda 1.0 \
   --loss-fd 1.0 \
   --loss-mfr 0.5 \
   --fda-k 6 \
   --fda-alpha 0.5 \
   --fd-margin 0.5