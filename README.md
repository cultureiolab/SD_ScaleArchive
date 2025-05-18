# SD Archive Project
This repository contains the tools used in Culture I/O Lab's large-scale generative visual media archival project. 

## ComfyUI Workflow
To properly initialize the workflow, ensure that ComfyUI Manager is installed, by cloning its repository to the custom_nodes folder of ComfyUI’s repository. Then, download WAS Node Suite, comfyui-yan, ComfyUI-Custom-Scripts, and ComfyUI-Logic. Once these steps are complete, uploading the workflow will populate all relevant nodes. 

The most relevant controls, and their uses, are as follows: 

* **Logic Boolean**: Set Boolean to 1 to reset the counter.
* **Text Load Line From File**: Set file_path to the location of the text file that contains prompts for archival, with each prompt separated by a line break. 
* **Load Checkpoint**: Set ckpt_name to the correct generative model. ComfyUI allows users to download and manage models through its Model Manager. Alternatively, models can be cloned directly into models/checkpoints within the ComfyUI repository.
* **TripleCLIPLoader**: For SD3.5, the TripleCLIPLoader is required. Download clip_l.safetensors, clip_g.safetensors, and t5xxl_fp16.safetensors. Link CLIP from the TripleCLIPLoader to the two CLIP Text Encoders. (Detach CLIP from Load Checkpoint)
* **Empty Latent Image**: Adjust dimensions with width and height, as well as batch size with the batch_size field. Note that large batches can take up significant memory.
* **Image Save**: Set output_path to output directory. 

## Composite Portrait Creation
The pipeline.py script creates composite portraits from the output of the above ComfyUI workflow (with supporting modules face.py and overlay.py). The script creates folders for each unique prompt in the title.txt file. Then, it processes all images in the IMAGE_ROOTS directories (delimit by commas) and moves them to the corresponding folders. Then, it runs the face.py and overlay.py scripts for each folder. The face.py script uses mediapipe package to detect facial features and identifies facial orientation using the nose-to-eyes ratio. After filtering out images with no identifiable facial features and those without front-orientation, the script anchores the eyes of the figure to two preset locations, cropping, rotating, and scaling photos in the process. The overlay.py script calculates the median pixel value for every pixel across all portraits, generating a composite image. 

To run this pipeline:
1. Install the required packages in requirements.txt.
2. Create a .txt file in the script directory that includes each prompt/keyword in a new line.
3. Update pipeline.py TITLE_FILE with the name of the aformentioned .txt file.
4. Update IMAGE_ROOTS with the folders where the images are located.
5. Run pipeline.py. 

## LAION Retrieval
This script retrieves images from the LAION-2B dataset, an open-source collection of web images. Given a keyword, it pulls and downloads relevant images that have a high aesthetic score, relevancy, as well as low watermark and NSFW probabilities. 

To run this script
1. Install the required packages in requirements.txt.
2. Replace word_pattern with your keyword.
3. Adjust banned_keywords to exclude any keywords/concepts.
4. Adjust watermark, similarity, nsfw parameters as needed.
5. Run laion.py


