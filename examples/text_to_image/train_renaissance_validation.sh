#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=rs_sd
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

cp /scratch/s1889338/all_captions/2500/metadata_2500_renaissance/metadata_best.jsonl /scratch/s1889338/all_captions/renaissance_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/renaissance_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite intricate and detailed. The painting features a mountainous landscape with a village nestled in the valley below. The village is surrounded by trees, and the sky is filled with clouds, creating a dramatic and picturesque scene. The artist has skillfully captured the natural elements of the landscape, such as the mountains, trees, and clouds, while also incorporating the village and its inhabitants. The overall composition of the painting is visually appealing and captures the viewer's attention, showcasing the artist's talent and the beauty of the natural world." \
  "The painting's composition is well-balanced, with a harmonious arrangement of elements. The cityscape, which includes buildings, boats, and people, is the central focus of the image. The boats are scattered throughout the scene, with some closer to the foreground and others further away, creating depth and dimension. The people are also dispersed throughout the painting, adding life and activity to the scene. The presence of mountains in the background provides a sense of scale and perspective, while the sky serves as a unifying element that connects all the elements together. Overall, the balance between the elements in the painting creates a visually appealing and engaging composition." \
  "In the painting, the movement is depicted through the use of a variety of elements, such as the tall grass, the bird, and the water. The tall grass and the bird in flight create a sense of movement and dynamism, as they both convey a sense of freedom and motion. The water, with its ripples and reflections, adds to the overall sense of movement and fluidity in the scene. The combination of these elements creates a visually engaging and dynamic composition that captures the viewer's attention and evokes a sense of motion and life." \
  "The focus point of this painting appears to be the central area where the angel is hovering over the head of the man in red. This area is illuminated, drawing the viewer's attention, and contrasts with the darker, more somber tones of the surrounding landscape." \
  "The painting employs a contrast between the ethereal and the earthly. The angel, with her heavenly wings and serene expression, stands in stark contrast to the earthly realm below. The rugged landscape, with its rocky outcrops and serene waters, further emphasizes this contrast. The angel's white robe contrasts with the rich reds of the other figures, drawing attention to her as the central figure." \
  "The painting employs a balanced composition. The river occupies the left half, providing a sense of depth and leading the viewer's eye towards the town. The town itself is symmetrically placed, with the church spire acting as a focal point. The mountains in the background provide a sense of scale and distance. The sky, though occupying a smaller portion, adds a dynamic element with its cloud formations." \
  "The foreground of the image is dominated by the two main figures, making them the primary focus. The background, on the other hand, provides context and depth to the scene. It showcases a landscape with buildings, suggesting a town or city, and a dramatic sky, possibly indicating an impending event or the end of a day. This juxtaposition creates a sense of depth and draws the viewer's attention to the central narrative while also providing a broader context to the story." \
  "The composition in the image is symmetrical. This can be determined by observing the two figures on either side of the central arch, which are mirror images of each other. Both figures are depicted in a similar pose, with one figure seated and the other standing, holding a musical instrument. The positioning of the figures, their attire, and the objects they hold are identical on both sides, which creates a sense of balance and symmetry. Additionally, the background architecture, which includes the arches and the wall behind the figures, is also symmetrical, reinforcing the overall symmetry of the composition. The symmetry is further emphasized by the presence of a bird in the upper left corner of the left panel, which is mirrored by the presence of a similar bird in the upper right corner of the right panel. This repetition of the bird adds to the harmony of the symmetrical design." \
  "In this composition, the viewer's eye movement is guided by the leading lines and the arrangement of the buildings. The tall buildings on the left side of the image have a steeple and a staircase, which draws the viewer's attention upward and to the left. The buildings on the right side have a more uniform height and are spaced further apart, creating a sense of depth and perspective. The central archway in the middle of the image acts as a focal point, leading the viewer's eye towards it and inviting them to imagine what lies beyond. The overall layout of the buildings and the use of light and shadow also contribute to a sense of depth and dimension, guiding the viewer's eye through the scene." \
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
  --output_dir="renaissance-best-35k" \
  --push_to_hub \
  --checkpointing_steps=36000

  
cp /scratch/s1889338/all_captions/2500/metadata_2500_renaissance/metadata_llava.jsonl /scratch/s1889338/all_captions/renaissance_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/renaissance_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite intricate and detailed. The painting features a mountainous landscape with a village nestled in the valley below. The village is surrounded by trees, and the sky is filled with clouds, creating a dramatic and picturesque scene. The artist has skillfully captured the natural elements of the landscape, such as the mountains, trees, and clouds, while also incorporating the village and its inhabitants. The overall composition of the painting is visually appealing and captures the viewer's attention, showcasing the artist's talent and the beauty of the natural world." \
  "The painting's composition is well-balanced, with a harmonious arrangement of elements. The cityscape, which includes buildings, boats, and people, is the central focus of the image. The boats are scattered throughout the scene, with some closer to the foreground and others further away, creating depth and dimension. The people are also dispersed throughout the painting, adding life and activity to the scene. The presence of mountains in the background provides a sense of scale and perspective, while the sky serves as a unifying element that connects all the elements together. Overall, the balance between the elements in the painting creates a visually appealing and engaging composition." \
  "In the painting, the movement is depicted through the use of a variety of elements, such as the tall grass, the bird, and the water. The tall grass and the bird in flight create a sense of movement and dynamism, as they both convey a sense of freedom and motion. The water, with its ripples and reflections, adds to the overall sense of movement and fluidity in the scene. The combination of these elements creates a visually engaging and dynamic composition that captures the viewer's attention and evokes a sense of motion and life." \
  "The focus point of this painting appears to be the central area where the angel is hovering over the head of the man in red. This area is illuminated, drawing the viewer's attention, and contrasts with the darker, more somber tones of the surrounding landscape." \
  "The painting employs a contrast between the ethereal and the earthly. The angel, with her heavenly wings and serene expression, stands in stark contrast to the earthly realm below. The rugged landscape, with its rocky outcrops and serene waters, further emphasizes this contrast. The angel's white robe contrasts with the rich reds of the other figures, drawing attention to her as the central figure." \
  "The painting employs a balanced composition. The river occupies the left half, providing a sense of depth and leading the viewer's eye towards the town. The town itself is symmetrically placed, with the church spire acting as a focal point. The mountains in the background provide a sense of scale and distance. The sky, though occupying a smaller portion, adds a dynamic element with its cloud formations." \
  "The foreground of the image is dominated by the two main figures, making them the primary focus. The background, on the other hand, provides context and depth to the scene. It showcases a landscape with buildings, suggesting a town or city, and a dramatic sky, possibly indicating an impending event or the end of a day. This juxtaposition creates a sense of depth and draws the viewer's attention to the central narrative while also providing a broader context to the story." \
  "The composition in the image is symmetrical. This can be determined by observing the two figures on either side of the central arch, which are mirror images of each other. Both figures are depicted in a similar pose, with one figure seated and the other standing, holding a musical instrument. The positioning of the figures, their attire, and the objects they hold are identical on both sides, which creates a sense of balance and symmetry. Additionally, the background architecture, which includes the arches and the wall behind the figures, is also symmetrical, reinforcing the overall symmetry of the composition. The symmetry is further emphasized by the presence of a bird in the upper left corner of the left panel, which is mirrored by the presence of a similar bird in the upper right corner of the right panel. This repetition of the bird adds to the harmony of the symmetrical design." \
  "In this composition, the viewer's eye movement is guided by the leading lines and the arrangement of the buildings. The tall buildings on the left side of the image have a steeple and a staircase, which draws the viewer's attention upward and to the left. The buildings on the right side have a more uniform height and are spaced further apart, creating a sense of depth and perspective. The central archway in the middle of the image acts as a focal point, leading the viewer's eye towards it and inviting them to imagine what lies beyond. The overall layout of the buildings and the use of light and shadow also contribute to a sense of depth and dimension, guiding the viewer's eye through the scene." \
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
  --output_dir="renaissance-llava-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_renaissance/metadata_cogvlm.jsonl /scratch/s1889338/all_captions/renaissance_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/renaissance_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite intricate and detailed. The painting features a mountainous landscape with a village nestled in the valley below. The village is surrounded by trees, and the sky is filled with clouds, creating a dramatic and picturesque scene. The artist has skillfully captured the natural elements of the landscape, such as the mountains, trees, and clouds, while also incorporating the village and its inhabitants. The overall composition of the painting is visually appealing and captures the viewer's attention, showcasing the artist's talent and the beauty of the natural world." \
  "The painting's composition is well-balanced, with a harmonious arrangement of elements. The cityscape, which includes buildings, boats, and people, is the central focus of the image. The boats are scattered throughout the scene, with some closer to the foreground and others further away, creating depth and dimension. The people are also dispersed throughout the painting, adding life and activity to the scene. The presence of mountains in the background provides a sense of scale and perspective, while the sky serves as a unifying element that connects all the elements together. Overall, the balance between the elements in the painting creates a visually appealing and engaging composition." \
  "In the painting, the movement is depicted through the use of a variety of elements, such as the tall grass, the bird, and the water. The tall grass and the bird in flight create a sense of movement and dynamism, as they both convey a sense of freedom and motion. The water, with its ripples and reflections, adds to the overall sense of movement and fluidity in the scene. The combination of these elements creates a visually engaging and dynamic composition that captures the viewer's attention and evokes a sense of motion and life." \
  "The focus point of this painting appears to be the central area where the angel is hovering over the head of the man in red. This area is illuminated, drawing the viewer's attention, and contrasts with the darker, more somber tones of the surrounding landscape." \
  "The painting employs a contrast between the ethereal and the earthly. The angel, with her heavenly wings and serene expression, stands in stark contrast to the earthly realm below. The rugged landscape, with its rocky outcrops and serene waters, further emphasizes this contrast. The angel's white robe contrasts with the rich reds of the other figures, drawing attention to her as the central figure." \
  "The painting employs a balanced composition. The river occupies the left half, providing a sense of depth and leading the viewer's eye towards the town. The town itself is symmetrically placed, with the church spire acting as a focal point. The mountains in the background provide a sense of scale and distance. The sky, though occupying a smaller portion, adds a dynamic element with its cloud formations." \
  "The foreground of the image is dominated by the two main figures, making them the primary focus. The background, on the other hand, provides context and depth to the scene. It showcases a landscape with buildings, suggesting a town or city, and a dramatic sky, possibly indicating an impending event or the end of a day. This juxtaposition creates a sense of depth and draws the viewer's attention to the central narrative while also providing a broader context to the story." \
  "The composition in the image is symmetrical. This can be determined by observing the two figures on either side of the central arch, which are mirror images of each other. Both figures are depicted in a similar pose, with one figure seated and the other standing, holding a musical instrument. The positioning of the figures, their attire, and the objects they hold are identical on both sides, which creates a sense of balance and symmetry. Additionally, the background architecture, which includes the arches and the wall behind the figures, is also symmetrical, reinforcing the overall symmetry of the composition. The symmetry is further emphasized by the presence of a bird in the upper left corner of the left panel, which is mirrored by the presence of a similar bird in the upper right corner of the right panel. This repetition of the bird adds to the harmony of the symmetrical design." \
  "In this composition, the viewer's eye movement is guided by the leading lines and the arrangement of the buildings. The tall buildings on the left side of the image have a steeple and a staircase, which draws the viewer's attention upward and to the left. The buildings on the right side have a more uniform height and are spaced further apart, creating a sense of depth and perspective. The central archway in the middle of the image acts as a focal point, leading the viewer's eye towards it and inviting them to imagine what lies beyond. The overall layout of the buildings and the use of light and shadow also contribute to a sense of depth and dimension, guiding the viewer's eye through the scene." \
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
  --output_dir="renaissance-cogvlm-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


cp /scratch/s1889338/all_captions/2500/metadata_2500_renaissance/metadata_deepseek.jsonl /scratch/s1889338/all_captions/renaissance_2500/train/metadata.jsonl

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/scratch/s1889338/all_captions/renaissance_2500/train" \
  --caption_column="text" \
  --validation_prompts \
  "The composition of this work is quite intricate and detailed. The painting features a mountainous landscape with a village nestled in the valley below. The village is surrounded by trees, and the sky is filled with clouds, creating a dramatic and picturesque scene. The artist has skillfully captured the natural elements of the landscape, such as the mountains, trees, and clouds, while also incorporating the village and its inhabitants. The overall composition of the painting is visually appealing and captures the viewer's attention, showcasing the artist's talent and the beauty of the natural world." \
  "The painting's composition is well-balanced, with a harmonious arrangement of elements. The cityscape, which includes buildings, boats, and people, is the central focus of the image. The boats are scattered throughout the scene, with some closer to the foreground and others further away, creating depth and dimension. The people are also dispersed throughout the painting, adding life and activity to the scene. The presence of mountains in the background provides a sense of scale and perspective, while the sky serves as a unifying element that connects all the elements together. Overall, the balance between the elements in the painting creates a visually appealing and engaging composition." \
  "In the painting, the movement is depicted through the use of a variety of elements, such as the tall grass, the bird, and the water. The tall grass and the bird in flight create a sense of movement and dynamism, as they both convey a sense of freedom and motion. The water, with its ripples and reflections, adds to the overall sense of movement and fluidity in the scene. The combination of these elements creates a visually engaging and dynamic composition that captures the viewer's attention and evokes a sense of motion and life." \
  "The focus point of this painting appears to be the central area where the angel is hovering over the head of the man in red. This area is illuminated, drawing the viewer's attention, and contrasts with the darker, more somber tones of the surrounding landscape." \
  "The painting employs a contrast between the ethereal and the earthly. The angel, with her heavenly wings and serene expression, stands in stark contrast to the earthly realm below. The rugged landscape, with its rocky outcrops and serene waters, further emphasizes this contrast. The angel's white robe contrasts with the rich reds of the other figures, drawing attention to her as the central figure." \
  "The painting employs a balanced composition. The river occupies the left half, providing a sense of depth and leading the viewer's eye towards the town. The town itself is symmetrically placed, with the church spire acting as a focal point. The mountains in the background provide a sense of scale and distance. The sky, though occupying a smaller portion, adds a dynamic element with its cloud formations." \
  "The foreground of the image is dominated by the two main figures, making them the primary focus. The background, on the other hand, provides context and depth to the scene. It showcases a landscape with buildings, suggesting a town or city, and a dramatic sky, possibly indicating an impending event or the end of a day. This juxtaposition creates a sense of depth and draws the viewer's attention to the central narrative while also providing a broader context to the story." \
  "The composition in the image is symmetrical. This can be determined by observing the two figures on either side of the central arch, which are mirror images of each other. Both figures are depicted in a similar pose, with one figure seated and the other standing, holding a musical instrument. The positioning of the figures, their attire, and the objects they hold are identical on both sides, which creates a sense of balance and symmetry. Additionally, the background architecture, which includes the arches and the wall behind the figures, is also symmetrical, reinforcing the overall symmetry of the composition. The symmetry is further emphasized by the presence of a bird in the upper left corner of the left panel, which is mirrored by the presence of a similar bird in the upper right corner of the right panel. This repetition of the bird adds to the harmony of the symmetrical design." \
  "In this composition, the viewer's eye movement is guided by the leading lines and the arrangement of the buildings. The tall buildings on the left side of the image have a steeple and a staircase, which draws the viewer's attention upward and to the left. The buildings on the right side have a more uniform height and are spaced further apart, creating a sense of depth and perspective. The central archway in the middle of the image acts as a focal point, leading the viewer's eye towards it and inviting them to imagine what lies beyond. The overall layout of the buildings and the use of light and shadow also contribute to a sense of depth and dimension, guiding the viewer's eye through the scene." \
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
  --output_dir="renaissance-deepseek-35k" \
  --push_to_hub \
  --checkpointing_steps=36000


python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-best-35k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-llava-35k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-cogvlm-35k --output_folder renaissance_2500_generations
python generate_images_from_prompts.py --js_file evaluation_renaissance_data.js --model_path /scratch/s1889338/diffusers/examples/text_to_image/renaissance-deepseek-35k --output_folder renaissance_2500_generations


deactivate
