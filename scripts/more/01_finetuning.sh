#!/bin/bash

export WANDB_PROJECT=Modality_Gap_v2
export TOKENIZER_PATH=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO



epochs=1
llama3_path=meta-llama/Meta-Llama-3.1-8B-Instruct
vicuna=lmsys/vicuna-7b-v1.5
images_path=../../MoE/playground/data
data_train_path=../../MoE/playground/data/llava_v1_5_mix625k.json
vision_tower=openai/clip-vit-large-patch14-336
mm_projector_path=ckpts/llava-llama-pretrain/mm_projector.bin

job_name="llava-llama-finetune-v2"
echo "job name: $job_name"

PORT=$((29500 + SLURM_JOB_ID % 1000))

deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $llama3_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version llama_3_1 \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--pretrain_mm_mlp_adapter $mm_projector_path \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./checkpoints/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 24000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to wandb \
--run_name $job_name \
