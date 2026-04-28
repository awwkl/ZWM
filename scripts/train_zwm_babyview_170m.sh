# Replicates the released BabyView-170M ZWM model (HF: awwkl/zwm-babyview-170m).
# 170M-parameter ZWM trained on BabyView 10s 256p clips, 512 global batch, 200k iters.
# Reference hardware: 8× NVIDIA A40 (48 GB), bfloat16.
# Edit --train_data_dir to point at your local copy of BabyView clips.

export SEED=$(date +%s)

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m zwm.train \
    --run_name zwm-babyview-170m \
    --model_config zwm.config.ZWM_170MConfig \
    --train_data_dir /path/to/babyview/clips/ \
    --dataloader_mode zwm_rgb_256 \
    --device cuda \
    --per_device_batch_size 16 \
    --batch_size 512 \
    --log_interval 10 \
    --frame1_mask_ratio 0.9 \
    --num_workers 16 \
    --accelerator_type A40 \
    --max_iters 200001 \
    --lr_decay_iters 200000 \
    --warmup_iters 2000 --warmdown_iters 198000 \
    --wandb \
    --save_interval 20000 \
    --dtype bfloat16 \
    --seed $SEED \
    --compile \
