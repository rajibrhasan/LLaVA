#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 25:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:6
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --output=log/slurm/modality_gap/log-%A-%a.log 
#SBATCH -J modality_gap

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export WANDB_API_KEY=

module purge
module load apptainer

PORT=$((29500 + SLURM_JOB_ID % 1000))

job_name="llava-v1.5-7b-pretrain"
echo "job name: $job_name"


apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v1.sif deepspeed --master_port $PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version plain \
    --data_path /scratch/mi8uu/mrm/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /scratch/mi8uu/mrm/LLaVA/playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ckpts/${job_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --diff_loss_coef 0.1 \
    --sim_loss_coef 0.1 \
    --report_to wandb \
    --run_name $job_name \


