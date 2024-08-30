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
  --validation_prompts \
  "In the painting, there is a contrast between the elements of the landscape and the trees. The landscape features a grassy field with a dirt road, while the trees are scattered throughout the scene. This contrast creates a sense of depth and balance in the composition, as the viewer can appreciate the different textures and colors of the grass, dirt, and trees. The presence of the trees also adds a sense of tranquility and natural beauty to the scene, further enhancing the overall aesthetic of the painting." \
  "In the painting, the proportion between the elements is well-balanced. The large mountain in the background is a dominant feature, taking up a significant portion of the image. The sunset adds warmth and depth to the scene, with the sun's rays illuminating the sky and casting a golden glow on the landscape. The presence of a boat on the water and a house nearby adds interest and variety to the composition. The boat and house are smaller in size compared to the mountain, but they still contribute to the overall balance and harmony of the painting. The artist has effectively combined these elements to create a visually appealing and engaging scene." \
  "In the image, the foreground and background are well-balanced, creating a harmonious composition. The foreground features a body of water with a sunset in the background, while the background includes a house and trees. The sunset adds warmth and depth to the scene, while the house and trees provide a sense of scale and context. The balance between the foreground and background elements allows the viewer to appreciate the beauty of the landscape and the tranquility of the scene." \
  "The composition is asymmetrical. While there is a balance in the distribution of trees and the path, the positioning of the two figures and the houses on the right side introduces an off-centered element, adding dynamism and interest to the scene." \
  "The viewer's eye movement is guided by the path of light, starting from the bright moon in the center, moving downwards to the reflection on the water, and then across the landscape to the silhouette of the trees and distant horizon. The composition uses the moon as a focal point, drawing the viewer's gaze inward and then outward to explore the surrounding environment." \
  "The painting employs a contrast between the elements to create depth and interest. The pathway, being the most prominent feature, draws the viewer's eye into the scene. The trees, with their varying shades of green, provide a contrast to the pathway and the sky. The distant figures add another layer of depth, making the scene appear more three-dimensional. The contrast between the darker foreground and the lighter background also enhances the overall mood and atmosphere of the painting." \
  "The focus point of this painting is the central area where the water lilies are most densely clustered. The artist has used a technique of applying paint in a thick, impasto style, which gives a sense of depth and volume to the water lilies. This technique, combined with the placement of the lilies in the center of the composition, draws the viewer's eye directly to this area. The water lilies are depicted with a high level of detail and realism, with the petals and leaves rendered in various shades of green, blue, and purple, which creates a visual contrast and draws attention to the individual elements. The use of color and the way the light reflects off the water surface enhances the three-dimensional effect, making the water lilies appear to be floating on the surface of the water. The surrounding area of the painting is less detailed, with softer brushstrokes and less defined forms, which creates a sense of depth and space." \
  "The painting depicts a landscape scene with a strong emphasis on the natural elements. The movement in this painting is primarily represented by the flowing lines and brushstrokes used to depict the trees and the sky. The trees, particularly the one in the foreground, are rendered with loose, expressive brushwork, suggesting the movement of the branches and leaves. The sky, visible in the background, is also painted with broad, sweeping strokes that convey a sense of depth and atmosphere. The ground is depicted with darker, more defined lines and shapes, contrasting with the lighter, more fluid depiction of the trees and sky. This contrast helps to anchor the scene and provide a sense of stability amidst the more dynamic elements. The overall impression is one of a serene, natural setting, where the movement of the elements—whether it be the wind in the trees or the clouds in the sky—is captured through the artist's choice of brushwork and color." \
  "The painting's composition exhibits a harmonious balance between the various elements. The central focus is the river, which is depicted with a rich array of blues and greens, suggesting the presence of water and reflections. The river is flanked by trees on both sides, which are rendered with a mix of green, blue, and yellow hues, creating a sense of depth and dimension. The sky, visible through the gaps in the foliage, adds a lighter, more neutral tone to the composition, providing a visual counterpoint to the denser colors of the trees and water. The presence of the two figures in a boat on the river adds a human element to the scene, providing a sense of scale and activity. The boat is positioned in the lower third of the painting, anchoring the composition and drawing the viewer's eye towards the center. The figures are small in comparison to the surrounding landscape, emphasizing the vastness of the natural environment." \
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
  --output_dir="landscape-best-35k" \
  --push_to_hub \
  --checkpointing_steps=36000

  
cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_llava.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "In the painting, there is a contrast between the elements of the landscape and the trees. The landscape features a grassy field with a dirt road, while the trees are scattered throughout the scene. This contrast creates a sense of depth and balance in the composition, as the viewer can appreciate the different textures and colors of the grass, dirt, and trees. The presence of the trees also adds a sense of tranquility and natural beauty to the scene, further enhancing the overall aesthetic of the painting." \
  "In the painting, the proportion between the elements is well-balanced. The large mountain in the background is a dominant feature, taking up a significant portion of the image. The sunset adds warmth and depth to the scene, with the sun's rays illuminating the sky and casting a golden glow on the landscape. The presence of a boat on the water and a house nearby adds interest and variety to the composition. The boat and house are smaller in size compared to the mountain, but they still contribute to the overall balance and harmony of the painting. The artist has effectively combined these elements to create a visually appealing and engaging scene." \
  "In the image, the foreground and background are well-balanced, creating a harmonious composition. The foreground features a body of water with a sunset in the background, while the background includes a house and trees. The sunset adds warmth and depth to the scene, while the house and trees provide a sense of scale and context. The balance between the foreground and background elements allows the viewer to appreciate the beauty of the landscape and the tranquility of the scene." \
  "The composition is asymmetrical. While there is a balance in the distribution of trees and the path, the positioning of the two figures and the houses on the right side introduces an off-centered element, adding dynamism and interest to the scene." \
  "The viewer's eye movement is guided by the path of light, starting from the bright moon in the center, moving downwards to the reflection on the water, and then across the landscape to the silhouette of the trees and distant horizon. The composition uses the moon as a focal point, drawing the viewer's gaze inward and then outward to explore the surrounding environment." \
  "The painting employs a contrast between the elements to create depth and interest. The pathway, being the most prominent feature, draws the viewer's eye into the scene. The trees, with their varying shades of green, provide a contrast to the pathway and the sky. The distant figures add another layer of depth, making the scene appear more three-dimensional. The contrast between the darker foreground and the lighter background also enhances the overall mood and atmosphere of the painting." \
  "The focus point of this painting is the central area where the water lilies are most densely clustered. The artist has used a technique of applying paint in a thick, impasto style, which gives a sense of depth and volume to the water lilies. This technique, combined with the placement of the lilies in the center of the composition, draws the viewer's eye directly to this area. The water lilies are depicted with a high level of detail and realism, with the petals and leaves rendered in various shades of green, blue, and purple, which creates a visual contrast and draws attention to the individual elements. The use of color and the way the light reflects off the water surface enhances the three-dimensional effect, making the water lilies appear to be floating on the surface of the water. The surrounding area of the painting is less detailed, with softer brushstrokes and less defined forms, which creates a sense of depth and space." \
  "The painting depicts a landscape scene with a strong emphasis on the natural elements. The movement in this painting is primarily represented by the flowing lines and brushstrokes used to depict the trees and the sky. The trees, particularly the one in the foreground, are rendered with loose, expressive brushwork, suggesting the movement of the branches and leaves. The sky, visible in the background, is also painted with broad, sweeping strokes that convey a sense of depth and atmosphere. The ground is depicted with darker, more defined lines and shapes, contrasting with the lighter, more fluid depiction of the trees and sky. This contrast helps to anchor the scene and provide a sense of stability amidst the more dynamic elements. The overall impression is one of a serene, natural setting, where the movement of the elements—whether it be the wind in the trees or the clouds in the sky—is captured through the artist's choice of brushwork and color." \
  "The painting's composition exhibits a harmonious balance between the various elements. The central focus is the river, which is depicted with a rich array of blues and greens, suggesting the presence of water and reflections. The river is flanked by trees on both sides, which are rendered with a mix of green, blue, and yellow hues, creating a sense of depth and dimension. The sky, visible through the gaps in the foliage, adds a lighter, more neutral tone to the composition, providing a visual counterpoint to the denser colors of the trees and water. The presence of the two figures in a boat on the river adds a human element to the scene, providing a sense of scale and activity. The boat is positioned in the lower third of the painting, anchoring the composition and drawing the viewer's eye towards the center. The figures are small in comparison to the surrounding landscape, emphasizing the vastness of the natural environment." \
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
  --output_dir="landscape-llava-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_cogvlm.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "In the painting, there is a contrast between the elements of the landscape and the trees. The landscape features a grassy field with a dirt road, while the trees are scattered throughout the scene. This contrast creates a sense of depth and balance in the composition, as the viewer can appreciate the different textures and colors of the grass, dirt, and trees. The presence of the trees also adds a sense of tranquility and natural beauty to the scene, further enhancing the overall aesthetic of the painting." \
  "In the painting, the proportion between the elements is well-balanced. The large mountain in the background is a dominant feature, taking up a significant portion of the image. The sunset adds warmth and depth to the scene, with the sun's rays illuminating the sky and casting a golden glow on the landscape. The presence of a boat on the water and a house nearby adds interest and variety to the composition. The boat and house are smaller in size compared to the mountain, but they still contribute to the overall balance and harmony of the painting. The artist has effectively combined these elements to create a visually appealing and engaging scene." \
  "In the image, the foreground and background are well-balanced, creating a harmonious composition. The foreground features a body of water with a sunset in the background, while the background includes a house and trees. The sunset adds warmth and depth to the scene, while the house and trees provide a sense of scale and context. The balance between the foreground and background elements allows the viewer to appreciate the beauty of the landscape and the tranquility of the scene." \
  "The composition is asymmetrical. While there is a balance in the distribution of trees and the path, the positioning of the two figures and the houses on the right side introduces an off-centered element, adding dynamism and interest to the scene." \
  "The viewer's eye movement is guided by the path of light, starting from the bright moon in the center, moving downwards to the reflection on the water, and then across the landscape to the silhouette of the trees and distant horizon. The composition uses the moon as a focal point, drawing the viewer's gaze inward and then outward to explore the surrounding environment." \
  "The painting employs a contrast between the elements to create depth and interest. The pathway, being the most prominent feature, draws the viewer's eye into the scene. The trees, with their varying shades of green, provide a contrast to the pathway and the sky. The distant figures add another layer of depth, making the scene appear more three-dimensional. The contrast between the darker foreground and the lighter background also enhances the overall mood and atmosphere of the painting." \
  "The focus point of this painting is the central area where the water lilies are most densely clustered. The artist has used a technique of applying paint in a thick, impasto style, which gives a sense of depth and volume to the water lilies. This technique, combined with the placement of the lilies in the center of the composition, draws the viewer's eye directly to this area. The water lilies are depicted with a high level of detail and realism, with the petals and leaves rendered in various shades of green, blue, and purple, which creates a visual contrast and draws attention to the individual elements. The use of color and the way the light reflects off the water surface enhances the three-dimensional effect, making the water lilies appear to be floating on the surface of the water. The surrounding area of the painting is less detailed, with softer brushstrokes and less defined forms, which creates a sense of depth and space." \
  "The painting depicts a landscape scene with a strong emphasis on the natural elements. The movement in this painting is primarily represented by the flowing lines and brushstrokes used to depict the trees and the sky. The trees, particularly the one in the foreground, are rendered with loose, expressive brushwork, suggesting the movement of the branches and leaves. The sky, visible in the background, is also painted with broad, sweeping strokes that convey a sense of depth and atmosphere. The ground is depicted with darker, more defined lines and shapes, contrasting with the lighter, more fluid depiction of the trees and sky. This contrast helps to anchor the scene and provide a sense of stability amidst the more dynamic elements. The overall impression is one of a serene, natural setting, where the movement of the elements—whether it be the wind in the trees or the clouds in the sky—is captured through the artist's choice of brushwork and color." \
  "The painting's composition exhibits a harmonious balance between the various elements. The central focus is the river, which is depicted with a rich array of blues and greens, suggesting the presence of water and reflections. The river is flanked by trees on both sides, which are rendered with a mix of green, blue, and yellow hues, creating a sense of depth and dimension. The sky, visible through the gaps in the foliage, adds a lighter, more neutral tone to the composition, providing a visual counterpoint to the denser colors of the trees and water. The presence of the two figures in a boat on the river adds a human element to the scene, providing a sense of scale and activity. The boat is positioned in the lower third of the painting, anchoring the composition and drawing the viewer's eye towards the center. The figures are small in comparison to the surrounding landscape, emphasizing the vastness of the natural environment." \
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
  --output_dir="landscape-cogvlm-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_landscape/metadata_deepseek.jsonl /scratch/s1889338/all_captions/landscape_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/landscape_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "In the painting, there is a contrast between the elements of the landscape and the trees. The landscape features a grassy field with a dirt road, while the trees are scattered throughout the scene. This contrast creates a sense of depth and balance in the composition, as the viewer can appreciate the different textures and colors of the grass, dirt, and trees. The presence of the trees also adds a sense of tranquility and natural beauty to the scene, further enhancing the overall aesthetic of the painting." \
  "In the painting, the proportion between the elements is well-balanced. The large mountain in the background is a dominant feature, taking up a significant portion of the image. The sunset adds warmth and depth to the scene, with the sun's rays illuminating the sky and casting a golden glow on the landscape. The presence of a boat on the water and a house nearby adds interest and variety to the composition. The boat and house are smaller in size compared to the mountain, but they still contribute to the overall balance and harmony of the painting. The artist has effectively combined these elements to create a visually appealing and engaging scene." \
  "In the image, the foreground and background are well-balanced, creating a harmonious composition. The foreground features a body of water with a sunset in the background, while the background includes a house and trees. The sunset adds warmth and depth to the scene, while the house and trees provide a sense of scale and context. The balance between the foreground and background elements allows the viewer to appreciate the beauty of the landscape and the tranquility of the scene." \
  "The composition is asymmetrical. While there is a balance in the distribution of trees and the path, the positioning of the two figures and the houses on the right side introduces an off-centered element, adding dynamism and interest to the scene." \
  "The viewer's eye movement is guided by the path of light, starting from the bright moon in the center, moving downwards to the reflection on the water, and then across the landscape to the silhouette of the trees and distant horizon. The composition uses the moon as a focal point, drawing the viewer's gaze inward and then outward to explore the surrounding environment." \
  "The painting employs a contrast between the elements to create depth and interest. The pathway, being the most prominent feature, draws the viewer's eye into the scene. The trees, with their varying shades of green, provide a contrast to the pathway and the sky. The distant figures add another layer of depth, making the scene appear more three-dimensional. The contrast between the darker foreground and the lighter background also enhances the overall mood and atmosphere of the painting." \
  "The focus point of this painting is the central area where the water lilies are most densely clustered. The artist has used a technique of applying paint in a thick, impasto style, which gives a sense of depth and volume to the water lilies. This technique, combined with the placement of the lilies in the center of the composition, draws the viewer's eye directly to this area. The water lilies are depicted with a high level of detail and realism, with the petals and leaves rendered in various shades of green, blue, and purple, which creates a visual contrast and draws attention to the individual elements. The use of color and the way the light reflects off the water surface enhances the three-dimensional effect, making the water lilies appear to be floating on the surface of the water. The surrounding area of the painting is less detailed, with softer brushstrokes and less defined forms, which creates a sense of depth and space." \
  "The painting depicts a landscape scene with a strong emphasis on the natural elements. The movement in this painting is primarily represented by the flowing lines and brushstrokes used to depict the trees and the sky. The trees, particularly the one in the foreground, are rendered with loose, expressive brushwork, suggesting the movement of the branches and leaves. The sky, visible in the background, is also painted with broad, sweeping strokes that convey a sense of depth and atmosphere. The ground is depicted with darker, more defined lines and shapes, contrasting with the lighter, more fluid depiction of the trees and sky. This contrast helps to anchor the scene and provide a sense of stability amidst the more dynamic elements. The overall impression is one of a serene, natural setting, where the movement of the elements—whether it be the wind in the trees or the clouds in the sky—is captured through the artist's choice of brushwork and color." \
  "The painting's composition exhibits a harmonious balance between the various elements. The central focus is the river, which is depicted with a rich array of blues and greens, suggesting the presence of water and reflections. The river is flanked by trees on both sides, which are rendered with a mix of green, blue, and yellow hues, creating a sense of depth and dimension. The sky, visible through the gaps in the foliage, adds a lighter, more neutral tone to the composition, providing a visual counterpoint to the denser colors of the trees and water. The presence of the two figures in a boat on the river adds a human element to the scene, providing a sense of scale and activity. The boat is positioned in the lower third of the painting, anchoring the composition and drawing the viewer's eye towards the center. The figures are small in comparison to the surrounding landscape, emphasizing the vastness of the natural environment." \
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
  --output_dir="landscape-deepseek-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-best-35k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-llava-35k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-cogvlm-35k --output_folder landscape_2500_generations
python generate_images_from_prompts.py --js_file evaluation_landscape_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/landscape-deepseek-35k --output_folder landscape_2500_generations

deactivate
