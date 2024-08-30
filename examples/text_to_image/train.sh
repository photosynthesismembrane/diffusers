#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=t2i
#SBATCH --mem=250G
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

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/diffusers/examples/text_to_image/renaissance_complete_augmented_subset_deepseek/train" \
  --caption_column="text" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="renaissance-deepseek-composition" \
  --push_to_hub \
  --checkpointing_steps=15000

deactivate
