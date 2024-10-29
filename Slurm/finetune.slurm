#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 1-10:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --output=log/slurm/modality_gap/log-%A-%a.log 
#SBATCH -J modality_gap

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export WANDB_PROJECT=Modality_Gap
export TOKENIZER_PATH=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning
export HF_HOME=
export WANDB_API_KEY=

module purge
module load apptainer

PORT=$((29500 + SLURM_JOB_ID % 1000))
epochs=1
llama3_path=meta-llama/Llama-3.1-8B-Instruct
vision_tower=openai/clip-vit-large-patch14-336
data_path=/scratch/mi8uu/mrm/LLaVA/playground/data/llava_v1_5_mix665k.json
image_folder=/scratch/mi8uu/mrm/LLaVA/playground/data
job_name="llava-llama-finetune"
pretrain_mm_mlp_adapter=ckpts/llava-llama-pretrain/mm_projector.bin
echo "job name: $job_name"



    apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v1.sif deepspeed --master_port $PORT llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $llama3_path \
    --version llama_3_1 \
    --llm_backbone llama_3_1 \
    --llm_pad_token pad \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --vision_tower $vision_tower \
    --pretrain_mm_mlp_adapter $pretrain_mm_mlp_adapter \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ckpts/${job_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --diff_loss_coef 100 \
    --sim_loss_coef 0.0 \
    --report_to wandb \
    --run_name ${job_name}
