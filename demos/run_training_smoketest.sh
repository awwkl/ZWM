# Smoke test for the ZWM training loop on bundled demo videos.
# Runs the 170M config for 301 iterations on a single GPU; saves checkpoints
# at iters 100, 200, 300 to out/zwm_smoketest/. Goal is to verify the full
# training pipeline runs end-to-end — NOT to produce a usable model.
# Reference hardware: 1× NVIDIA A40 (48 GB), bfloat16, ~14 GB peak VRAM, ~10–15 min.

export SEED=$(date +%s)

python -m zwm.train \
    --run_name zwm_smoketest \
    --model_config zwm.config.ZWM_170MConfig \
    --train_data_dir data/demo_videos/ \
    --dataloader_mode zwm_rgb_256 \
    --device cuda \
    --per_device_batch_size 8 \
    --batch_size 8 \
    --log_interval 10 \
    --frame1_mask_ratio 0.9 \
    --num_workers 2 \
    --accelerator_type A40 \
    --max_iters 301 \
    --lr_decay_iters 301 \
    --warmup_iters 50 --warmdown_iters 250 \
    --save_interval 100 \
    --dtype bfloat16 \
    --seed $SEED \
