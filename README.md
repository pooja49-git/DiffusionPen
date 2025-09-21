DiffusionPen for Hindi: A Handwriting Generation & Style Transfer Project
This project is an implementation of a conditional diffusion model, based on the "DiffusionPen" paper, specifically adapted and trained for generating handwritten text in Hindi. The system can learn a person's unique handwriting style from a few samples and generate new, unseen words and sentences in that style.

Key Features
High-Quality Hindi Word Generation: Creates clear, realistic images of handwritten Hindi words.

Few-Shot Style Transfer: Learns a unique handwriting style from just a small number of sample images (5-10).

Paragraph Generation: Generates individual words and intelligently stitches them together to form coherent, multi-line paragraphs.

Two-Stage Training Pipeline:

A Style Encoder is first trained to understand and quantify handwriting style.

A U-Net Generator is then trained to create images, guided by the pre-trained Style Encoder.

Custom Dataset Compatibility: The data loaders are built to work with a custom dataset structure defined by simple annotation files.

Generated Examples
The model was successfully trained on a custom dataset of ~70,000 Hindi word images from 7 different writers.

Single Word Generation
The model can generate different words while maintaining a consistent handwriting style.

Paragraph Generation
The system can generate and arrange multiple words to form a full paragraph.

System Architecture
The project uses a two-model system, as described in the DiffusionPen paper:

The Style Encoder (The "Art Critic"): A pre-trained MobileNetV2 model is fine-tuned using Triplet Loss. It learns to create a 1280-dimensional style vector that numerically represents a handwriting style.

The U-Net Generator (The "Artist"): This is the main diffusion model. It takes a random noise input and learns to denoise it into a clean image. Its work is guided by two conditions:

The style vector from the Style Encoder.

A text embedding (from a CANINE model) representing the word to be written.

Setup and Installation
Clone the repository:

Bash

git clone [Your-Repo-URL]
cd [Your-Repo-Name]
Create and activate a Conda environment:

Bash

conda create -n diffpen python=3.8
conda activate diffpen
Install dependencies:

Bash

pip install torch torchvision torchaudio
pip install transformers diffusers timm tqdm numpy Pillow
Download Pre-trained Models:

Run the provided helper scripts to download the CANINE Tokenizer & Model and the Stable Diffusion v1.5 components to their respective local folders.

Usage
This is a two-stage project. You must first train the Style Encoder, and then train the main Generator.

1. Prepare Your Dataset
Ensure your dataset is organized in the structure data_root/split/writer_id/.../image.jpg.

Create train.txt and val.txt annotation files where each line is: relative/path/to/image.jpg word

2. Train the Style Encoder
This script trains the model that learns handwriting styles.

Bash

python style_encoder_train_wordstylist.py \
    --train_annotation /path/to/your/train.txt \
    --val_annotation /path/to/your/val.txt \
    --data_root /path/to/your/data/ \
    --save_path ./style_models
3. Train the Main Generator
This script trains the U-Net that generates the images, using the style encoder you just trained.

Bash

sbatch run_training.sh
(Ensure the paths inside run_training.sh are correct, especially the --style_path which must point to the output of the previous step).

4. Generate New Handwriting (Inference)
After the main generator is trained, use this script to create new images.

Bash

python generate_sentence.py \
    --text "आपका स्वागत है" \
    --style_folder ./style_samples/ \
    --style_path ./style_models/wordstylist_mobilenetv2_100_best.pth \
    --save_path ./diffusion_models \
    --train_annotation /path/to/your/train.txt
Acknowledgments
This project is based on the methods and architecture described in the paper:

DiffusionPen: Towards Controlling the Style of Handwritten Text Generation by Nikolaidou, K., Retsinas, G., Sfikas, G., & Liwicki, M. (2024).







