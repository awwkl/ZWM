
export SEED=$(date +%s)

# CUDA_VISIBLE_DEVICES=6,7 \
# python \
    # --master_addr=10.102.2.158 \
    # --node_rank=$NODE_RANK \
    # --master_port=8327 \
# NODE_RANK=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    zwm/train.py \
    --run_name ZWM_1B_RGB_Babyview_2025.2_200k \
    --model_config zwm.config.ZWM_1BConfig \
    --train_data_dir /ccn2a/dataset/babyview/2025.2/split_10s_clips_256p/ \
    --dataloader_mode zwm_rgb_256 \
    --device cuda \
    --per_device_batch_size 8 \
    --batch_size 512 \
    --log_interval 10 \
    --frame1_mask_ratio 0.9 \
    --num_workers 16 \
    --accelerator_type A40 \
    --max_iters 200001 \
    --lr_decay_iters 200000 \
    --warmup_iters 2000 --warmdown_iters 198000 \
    --wandb \
    --save_interval 10000 \
    --dtype bfloat16 \
    --seed $SEED \
    --compile \
    # --resume_from "ZWM1B_RGB_Babyview_2025.2_200k/model_00010000.pt" \
    # --resume_optimizer \
    # --fsdp \
    # --fsdp \

    # --run_name ZWM100M_Patch8_RGBIMU_Babyview_gpu \
    # --model_config zwm.config.ZWM100M_Patch8_RGBIMUConfig \
    # --dataloader_mode rgb_imu_shuffle \

    # --save_interval 200 \
    # --wandb --wandb_org khaiaw-stanford-university --wandb_project zwm \

    # torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$NODE_RANK --master_addr=10.102.2.158 --master_port=8001 \

    # --train_data_dir /ccn2/dataset/babyview/quant_chunked \
    # --batch_size 512 --per_device_batch_size 4 --log_interval 1 \
    # --wandb \