#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=ab_sd
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

cp /scratch/s1889338/all_captions/2500/metadata_2500_abstract/metadata_best.jsonl /scratch/s1889338/all_captions/abstract_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/abstract_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite interesting and dynamic. The painting features a large, colorful landscape with a mix of orange, blue, and green hues. The artist has skillfully combined these colors to create a vibrant and lively scene. The painting also includes a mountainous area, which adds depth and complexity to the composition. The overall effect is a visually engaging and captivating piece of art that showcases the artist's talent and creativity." \
  "The painting features a variety of elements, including a black and blue background, a yellow and red foreground, and a multitude of colors and shapes in between. The balance between these elements is achieved through the use of contrasting colors and the arrangement of shapes. The black and blue background creates a sense of depth and contrasts with the vibrant colors of the foreground. The yellow and red colors in the foreground stand out and draw the viewer's attention, while the multitude of colors and shapes in between create a sense of visual interest and balance. Overall, the painting's composition effectively combines contrasting elements to create a visually engaging and balanced piece of art." \
  "The painting features a vibrant and dynamic composition, with a large, curved, and colorful line that dominates the image. This line, which appears to be a long, flowing, and swirling shape, adds a sense of movement and energy to the artwork. The combination of the bold colors and the fluid form creates a visually engaging and dynamic piece of art that captures the viewer's attention." \
  "The focus point of this painting appears to be the central area where the bright orange and blue splatters converge. This area draws the viewer's eye due to its contrasting colors and the dynamic interplay of the paint." \
  "The painting showcases a striking contrast between the elements. The central white area serves as a calm and neutral base, while the vibrant streaks of colors radiate outwards, creating a dynamic and energetic effect. This juxtaposition of calm and chaos adds depth and intrigue to the composition." \
  "The composition of the painting seems to prioritize balance and movement. The darker elements, possibly representing land or mountains, dominate the upper half, while the lighter elements, possibly representing water or sky, dominate the lower half. The interplay between these elements creates a dynamic visual tension, drawing the viewer's eye across the canvas." \
  "The foreground and background relationship in this composition is quite dynamic and layered. In the foreground, we see a series of abstract shapes and forms that appear to be floating or suspended in space. These shapes are primarily in shades of blue, with some white and black accents, and they are arranged in a way that suggests depth and movement. The blue shapes are varied in size and shape, with some appearing larger and more prominent, while others are smaller and more subtle. The background of the composition is dominated by a white space that contrasts with the blue shapes in the foreground. This white space is punctuated by a series of black and red shapes that seem to emerge from the background, adding a sense of depth and layering to the overall composition. These shapes are more abstract and less defined than the blue shapes in the foreground, which gives the impression that they are floating or suspended in the white space." \
  "The composition in the image is asymmetrical. This can be determined by observing the various shapes and forms that make up the artwork. The image shows a variety of irregular shapes and lines, with no clear mirror image or repetition that would indicate symmetry. For instance, the large black and white shapes on the left side of the image do not have a counterpart on the right side that would mirror their form. Similarly, the smaller shapes and lines scattered throughout the composition do not align in a way that suggests symmetry. The colors used, such as the yellow and red accents, are also distributed unevenly, further contributing to the asymmetrical nature of the piece. The visual clues that support this statement include the absence of repeated patterns or mirrored shapes, the varied sizes and orientations of the forms, and the distribution of colors that do not follow a consistent pattern. All these elements contribute to the overall impression of an asymmetrical composition." \
  "The viewer's eye movement in this composition is guided by the contrasting colors and the arrangement of the paint splatters. The image is dominated by a white background, which serves as a neutral canvas that allows the vibrant colors of the paint splatters to stand out. The most prominent feature is the large, central orange paint splatter, which draws the eye immediately due to its size and bright color. This central element is flanked by other splatters in various colors, including blue, purple, and red, which are arranged in a way that creates a sense of balance and movement. The blue and purple splatters are positioned to the left and right of the orange splatter, respectively, while the red splatter is placed above the orange one, creating a sort of triangular formation that guides the eye across the image. The paint splatters are not uniform in size, with some appearing larger and more central, while others are smaller and more peripheral." \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=35000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --validation_epochs=500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="abstract-best-35k" \
  --push_to_hub \
  --checkpointing_steps=36000

  
cp /scratch/s1889338/all_captions/2500/metadata_2500_abstract/metadata_llava.jsonl /scratch/s1889338/all_captions/abstract_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/abstract_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite interesting and dynamic. The painting features a large, colorful landscape with a mix of orange, blue, and green hues. The artist has skillfully combined these colors to create a vibrant and lively scene. The painting also includes a mountainous area, which adds depth and complexity to the composition. The overall effect is a visually engaging and captivating piece of art that showcases the artist's talent and creativity." \
  "The painting features a variety of elements, including a black and blue background, a yellow and red foreground, and a multitude of colors and shapes in between. The balance between these elements is achieved through the use of contrasting colors and the arrangement of shapes. The black and blue background creates a sense of depth and contrasts with the vibrant colors of the foreground. The yellow and red colors in the foreground stand out and draw the viewer's attention, while the multitude of colors and shapes in between create a sense of visual interest and balance. Overall, the painting's composition effectively combines contrasting elements to create a visually engaging and balanced piece of art." \
  "The painting features a vibrant and dynamic composition, with a large, curved, and colorful line that dominates the image. This line, which appears to be a long, flowing, and swirling shape, adds a sense of movement and energy to the artwork. The combination of the bold colors and the fluid form creates a visually engaging and dynamic piece of art that captures the viewer's attention." \
  "The focus point of this painting appears to be the central area where the bright orange and blue splatters converge. This area draws the viewer's eye due to its contrasting colors and the dynamic interplay of the paint." \
  "The painting showcases a striking contrast between the elements. The central white area serves as a calm and neutral base, while the vibrant streaks of colors radiate outwards, creating a dynamic and energetic effect. This juxtaposition of calm and chaos adds depth and intrigue to the composition." \
  "The composition of the painting seems to prioritize balance and movement. The darker elements, possibly representing land or mountains, dominate the upper half, while the lighter elements, possibly representing water or sky, dominate the lower half. The interplay between these elements creates a dynamic visual tension, drawing the viewer's eye across the canvas." \
  "The foreground and background relationship in this composition is quite dynamic and layered. In the foreground, we see a series of abstract shapes and forms that appear to be floating or suspended in space. These shapes are primarily in shades of blue, with some white and black accents, and they are arranged in a way that suggests depth and movement. The blue shapes are varied in size and shape, with some appearing larger and more prominent, while others are smaller and more subtle. The background of the composition is dominated by a white space that contrasts with the blue shapes in the foreground. This white space is punctuated by a series of black and red shapes that seem to emerge from the background, adding a sense of depth and layering to the overall composition. These shapes are more abstract and less defined than the blue shapes in the foreground, which gives the impression that they are floating or suspended in the white space." \
  "The composition in the image is asymmetrical. This can be determined by observing the various shapes and forms that make up the artwork. The image shows a variety of irregular shapes and lines, with no clear mirror image or repetition that would indicate symmetry. For instance, the large black and white shapes on the left side of the image do not have a counterpart on the right side that would mirror their form. Similarly, the smaller shapes and lines scattered throughout the composition do not align in a way that suggests symmetry. The colors used, such as the yellow and red accents, are also distributed unevenly, further contributing to the asymmetrical nature of the piece. The visual clues that support this statement include the absence of repeated patterns or mirrored shapes, the varied sizes and orientations of the forms, and the distribution of colors that do not follow a consistent pattern. All these elements contribute to the overall impression of an asymmetrical composition." \
  "The viewer's eye movement in this composition is guided by the contrasting colors and the arrangement of the paint splatters. The image is dominated by a white background, which serves as a neutral canvas that allows the vibrant colors of the paint splatters to stand out. The most prominent feature is the large, central orange paint splatter, which draws the eye immediately due to its size and bright color. This central element is flanked by other splatters in various colors, including blue, purple, and red, which are arranged in a way that creates a sense of balance and movement. The blue and purple splatters are positioned to the left and right of the orange splatter, respectively, while the red splatter is placed above the orange one, creating a sort of triangular formation that guides the eye across the image. The paint splatters are not uniform in size, with some appearing larger and more central, while others are smaller and more peripheral." \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=35000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --validation_epochs=500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="abstract-llava-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_abstract/metadata_cogvlm.jsonl /scratch/s1889338/all_captions/abstract_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/abstract_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite interesting and dynamic. The painting features a large, colorful landscape with a mix of orange, blue, and green hues. The artist has skillfully combined these colors to create a vibrant and lively scene. The painting also includes a mountainous area, which adds depth and complexity to the composition. The overall effect is a visually engaging and captivating piece of art that showcases the artist's talent and creativity." \
  "The painting features a variety of elements, including a black and blue background, a yellow and red foreground, and a multitude of colors and shapes in between. The balance between these elements is achieved through the use of contrasting colors and the arrangement of shapes. The black and blue background creates a sense of depth and contrasts with the vibrant colors of the foreground. The yellow and red colors in the foreground stand out and draw the viewer's attention, while the multitude of colors and shapes in between create a sense of visual interest and balance. Overall, the painting's composition effectively combines contrasting elements to create a visually engaging and balanced piece of art." \
  "The painting features a vibrant and dynamic composition, with a large, curved, and colorful line that dominates the image. This line, which appears to be a long, flowing, and swirling shape, adds a sense of movement and energy to the artwork. The combination of the bold colors and the fluid form creates a visually engaging and dynamic piece of art that captures the viewer's attention." \
  "The focus point of this painting appears to be the central area where the bright orange and blue splatters converge. This area draws the viewer's eye due to its contrasting colors and the dynamic interplay of the paint." \
  "The painting showcases a striking contrast between the elements. The central white area serves as a calm and neutral base, while the vibrant streaks of colors radiate outwards, creating a dynamic and energetic effect. This juxtaposition of calm and chaos adds depth and intrigue to the composition." \
  "The composition of the painting seems to prioritize balance and movement. The darker elements, possibly representing land or mountains, dominate the upper half, while the lighter elements, possibly representing water or sky, dominate the lower half. The interplay between these elements creates a dynamic visual tension, drawing the viewer's eye across the canvas." \
  "The foreground and background relationship in this composition is quite dynamic and layered. In the foreground, we see a series of abstract shapes and forms that appear to be floating or suspended in space. These shapes are primarily in shades of blue, with some white and black accents, and they are arranged in a way that suggests depth and movement. The blue shapes are varied in size and shape, with some appearing larger and more prominent, while others are smaller and more subtle. The background of the composition is dominated by a white space that contrasts with the blue shapes in the foreground. This white space is punctuated by a series of black and red shapes that seem to emerge from the background, adding a sense of depth and layering to the overall composition. These shapes are more abstract and less defined than the blue shapes in the foreground, which gives the impression that they are floating or suspended in the white space." \
  "The composition in the image is asymmetrical. This can be determined by observing the various shapes and forms that make up the artwork. The image shows a variety of irregular shapes and lines, with no clear mirror image or repetition that would indicate symmetry. For instance, the large black and white shapes on the left side of the image do not have a counterpart on the right side that would mirror their form. Similarly, the smaller shapes and lines scattered throughout the composition do not align in a way that suggests symmetry. The colors used, such as the yellow and red accents, are also distributed unevenly, further contributing to the asymmetrical nature of the piece. The visual clues that support this statement include the absence of repeated patterns or mirrored shapes, the varied sizes and orientations of the forms, and the distribution of colors that do not follow a consistent pattern. All these elements contribute to the overall impression of an asymmetrical composition." \
  "The viewer's eye movement in this composition is guided by the contrasting colors and the arrangement of the paint splatters. The image is dominated by a white background, which serves as a neutral canvas that allows the vibrant colors of the paint splatters to stand out. The most prominent feature is the large, central orange paint splatter, which draws the eye immediately due to its size and bright color. This central element is flanked by other splatters in various colors, including blue, purple, and red, which are arranged in a way that creates a sense of balance and movement. The blue and purple splatters are positioned to the left and right of the orange splatter, respectively, while the red splatter is placed above the orange one, creating a sort of triangular formation that guides the eye across the image. The paint splatters are not uniform in size, with some appearing larger and more central, while others are smaller and more peripheral." \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=35000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --validation_epochs=500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="abstract-cogvlm-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_abstract/metadata_deepseek.jsonl /scratch/s1889338/all_captions/abstract_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/abstract_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite interesting and dynamic. The painting features a large, colorful landscape with a mix of orange, blue, and green hues. The artist has skillfully combined these colors to create a vibrant and lively scene. The painting also includes a mountainous area, which adds depth and complexity to the composition. The overall effect is a visually engaging and captivating piece of art that showcases the artist's talent and creativity." \
  "The painting features a variety of elements, including a black and blue background, a yellow and red foreground, and a multitude of colors and shapes in between. The balance between these elements is achieved through the use of contrasting colors and the arrangement of shapes. The black and blue background creates a sense of depth and contrasts with the vibrant colors of the foreground. The yellow and red colors in the foreground stand out and draw the viewer's attention, while the multitude of colors and shapes in between create a sense of visual interest and balance. Overall, the painting's composition effectively combines contrasting elements to create a visually engaging and balanced piece of art." \
  "The painting features a vibrant and dynamic composition, with a large, curved, and colorful line that dominates the image. This line, which appears to be a long, flowing, and swirling shape, adds a sense of movement and energy to the artwork. The combination of the bold colors and the fluid form creates a visually engaging and dynamic piece of art that captures the viewer's attention." \
  "The focus point of this painting appears to be the central area where the bright orange and blue splatters converge. This area draws the viewer's eye due to its contrasting colors and the dynamic interplay of the paint." \
  "The painting showcases a striking contrast between the elements. The central white area serves as a calm and neutral base, while the vibrant streaks of colors radiate outwards, creating a dynamic and energetic effect. This juxtaposition of calm and chaos adds depth and intrigue to the composition." \
  "The composition of the painting seems to prioritize balance and movement. The darker elements, possibly representing land or mountains, dominate the upper half, while the lighter elements, possibly representing water or sky, dominate the lower half. The interplay between these elements creates a dynamic visual tension, drawing the viewer's eye across the canvas." \
  "The foreground and background relationship in this composition is quite dynamic and layered. In the foreground, we see a series of abstract shapes and forms that appear to be floating or suspended in space. These shapes are primarily in shades of blue, with some white and black accents, and they are arranged in a way that suggests depth and movement. The blue shapes are varied in size and shape, with some appearing larger and more prominent, while others are smaller and more subtle. The background of the composition is dominated by a white space that contrasts with the blue shapes in the foreground. This white space is punctuated by a series of black and red shapes that seem to emerge from the background, adding a sense of depth and layering to the overall composition. These shapes are more abstract and less defined than the blue shapes in the foreground, which gives the impression that they are floating or suspended in the white space." \
  "The composition in the image is asymmetrical. This can be determined by observing the various shapes and forms that make up the artwork. The image shows a variety of irregular shapes and lines, with no clear mirror image or repetition that would indicate symmetry. For instance, the large black and white shapes on the left side of the image do not have a counterpart on the right side that would mirror their form. Similarly, the smaller shapes and lines scattered throughout the composition do not align in a way that suggests symmetry. The colors used, such as the yellow and red accents, are also distributed unevenly, further contributing to the asymmetrical nature of the piece. The visual clues that support this statement include the absence of repeated patterns or mirrored shapes, the varied sizes and orientations of the forms, and the distribution of colors that do not follow a consistent pattern. All these elements contribute to the overall impression of an asymmetrical composition." \
  "The viewer's eye movement in this composition is guided by the contrasting colors and the arrangement of the paint splatters. The image is dominated by a white background, which serves as a neutral canvas that allows the vibrant colors of the paint splatters to stand out. The most prominent feature is the large, central orange paint splatter, which draws the eye immediately due to its size and bright color. This central element is flanked by other splatters in various colors, including blue, purple, and red, which are arranged in a way that creates a sense of balance and movement. The blue and purple splatters are positioned to the left and right of the orange splatter, respectively, while the red splatter is placed above the orange one, creating a sort of triangular formation that guides the eye across the image. The paint splatters are not uniform in size, with some appearing larger and more central, while others are smaller and more peripheral." \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=35000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --validation_epochs=500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="abstract-deepseek-35k" \
  --push_to_hub \
  --checkpointing_steps=36000

python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-best-35k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-llava-35k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-cogvlm-35k --output_folder abstract_2500_generations
python generate_images_from_prompts.py --js_file evaluation_abstract_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/abstract-deepseek-35k --output_folder abstract_2500_generations

deactivate
