#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=ls_sd
#SBATCH --mem=100G
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load GCCcore/12.3.0

source ../../venv/bin/activate

export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export HF_HOME="/scratch/$USER/.cache/huggingface/transformers"

accelerate config default

huggingface-cli login --token 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

ulimit -s unlimited

cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_best.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="landscape-best-8k" \
  --push_to_hub \
  --checkpointing_steps=8000

  
cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_llava.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="landscape-llava-8k" \
  --push_to_hub \
  --checkpointing_steps=8000


cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_cogvlm.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="landscape-cogvlm-8k" \
  --push_to_hub \
  --checkpointing_steps=8000


cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_deepseek.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="landscape-deepseek-8k" \
  --push_to_hub \
  --checkpointing_steps=8000


python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-best-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-llava-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-cogvlm-8k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-deepseek-8k --output_folder landscape_2500_generations

deactivate
