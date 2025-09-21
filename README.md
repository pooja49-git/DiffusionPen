# DiffusionPen for Hindi: A Handwriting Generation & Style Transfer Project

This project implements a conditional diffusion model, based on the "DiffusionPen" paper, specifically adapted and trained for generating handwritten text in **Hindi**. The system learns a person's unique handwriting style from a few samples and synthesizes new, unseen words and sentences in that precise style.

## Key Features

* **High-Quality Hindi Text Generation:** Produces clear, realistic images of handwritten Hindi words and sentences.
* **Few-Shot Style Transfer:** Accurately replicates and transfers a unique handwriting style from just a small set of example images (typically 5-10).
* **Paragraph Synthesis:** Capable of generating individual words and intelligently composing them into coherent, multi-line paragraphs.
* **Robust Two-Stage Training Pipeline:**
    * A **Style Encoder** is initially trained to understand and quantify handwriting style using triplet loss.
    * A **U-Net Generator** is subsequently trained to produce images, conditioned by the pre-trained Style Encoder and text embeddings.
* **Custom Dataset Compatibility:** Designed with flexible data loaders to integrate custom Hindi handwriting datasets via simple annotation files.

## Generated Examples

The model was successfully trained on a custom dataset comprising approximately **70,000 Hindi word images** from **7 distinct writers**.

### Single Word Generation
<img width="1034" height="68" alt="image" src="https://github.com/user-attachments/assets/1f6de2da-8bbe-416b-ab23-95efb3841523" />


### Paragraph Generation
<img width="1024" height="1932" alt="image" src="https://github.com/user-attachments/assets/61482e35-ad93-4acf-88a2-0b47c24b6a90" />


## System Architecture

The project employs a robust two-model system, mirroring the architecture proposed in the DiffusionPen paper:

1.  **The Style Encoder (The "Handwriting Analyst"):**
    * Utilizes a pre-trained **MobileNetV2** backbone, fine-tuned using **Triplet Loss**.
    * Its function is to extract a unique **1280-dimensional style vector** from handwriting samples, mathematically representing a writer's style.

2.  **The U-Net Generator (The "Digital Scribe"):**
    * The core diffusion model responsible for image generation.
    * It progressively denoises a random noise input into a coherent handwriting image.
    * Its generation process is critically conditioned by two inputs:
        * The **style vector** provided by the Style Encoder.
        * A **text embedding** (derived from a `CANINE` model) representing the textual content to be generated.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL_HERE]
    cd [YOUR_REPO_NAME]
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n diffpen_hindi python=3.8
    conda activate diffpen_hindi
    ```

3.  **Install project dependencies:**
    ```bash
    pip install torch torchvision torchaudio transformers diffusers timm tqdm numpy Pillow
    ```

4.  **Download Pre-trained Models:**
    The project relies on a few pre-trained models. These should be downloaded and placed in specific directories.

    * **CANINE Tokenizer & Model:** Used for text embeddings.
        * You can typically download these through the `transformers` library when first used, or manually from Hugging Face: `google/canine-c`. Ensure they are accessible by your script (e.g., in a `./canine_model/` directory if your code expects it there).
    * **Stable Diffusion v1.5 components (VAE, U-Net, Scheduler):** These form the base of the image generation pipeline.
        * These can be downloaded using the `diffusers` library. Your code might have a function for this, or you might need to manually download them to a `./stable_diffusion_weights/` directory (or similar path expected by your code).

    *(**Note:** Specific download instructions or helper scripts should be referenced here if they exist in your repository.)*

## Usage Guide

This project follows a two-stage training process, followed by inference (generation).

### 1. Prepare Your Dataset

* Organize your handwriting image data with the structure: `data_root/split/writer_id/image.jpg`.
* Create `train.txt` and `val.txt` annotation files, where each line specifies: `relative/path/to/image.jpg word_transcription`.
    * Example `train.txt` entry: `writer001/img_0001.jpg आपका`

### 2. Train the Style Encoder

This script trains the `ImageEncoder` to learn distinct handwriting styles using triplet loss.
The output will be saved to `./style_models/wordstylist_mobilenetv2_100_best.pth`.

```bash
python style_encoder_train_wordstylist.py \
    --train_annotation /path/to/your/train.txt \
    --val_annotation /path/to/your/val.txt \
    --data_root /path/to/your/data/ \
    --save_path ./style_models \
    --num_epochs 100 \
    --batch_size 64 \
    --lr 0.0001
    # Add other parameters you used for your training
```

### 3\. Train the Main Generator (U-Net)

This script trains the U-Net for image generation, critically utilizing the style encoder trained in the previous step.

```bash
# If using a Slurm cluster, you might run:
# sbatch run_training.sh

# For direct execution, use a command similar to this (adjust parameters as needed):
python train_wordstylist.py \
    --train_annotation /path/to/your/train.txt \
    --val_annotation /path/to/your/val.txt \
    --data_root /path/to/your/data/ \
    --style_path ./style_models/wordstylist_mobilenetv2_100_best.pth \
    --save_path ./diffusion_models \
    --num_epochs 1000 \
    --batch_size 16 \
    --lr 0.00001 \
    --image_size 128 \
    --gpu_ids 0 \
    --project_name diffpen_hindi_generator
    # ... include all other relevant parameters used in your actual training run.
```

*(**Note:** `--style_path` must correctly point to the `wordstylist_mobilenetv2_100_best.pth` file generated from Step 2. `--ema_path` in `generate_sentence.py` will typically point to `diffusion_models/ema_ckpt.pt` which is the output of this training step.)*

### 4\. Generate New Handwriting (Inference)

After the main generator is trained, use this script to synthesize new handwriting images from text and style samples.

```bash
python generate_sentence.py \
    --text "आपका स्वागत है" \
    --style_folder ./path/to/your/style_sample_images/ \
    --style_path ./style_models/wordstylist_mobilenetv2_100_best.pth \
    --ema_path ./diffusion_models/ema_ckpt.pt \
    --save_path ./output_images \
    --train_annotation /path/to/your/train.txt \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --num_samples 1
    # ... any other parameters relevant for generation
```

*(**Note:** `--style_folder` should contain 5-10 *distinct* images from a single writer whose style you want to replicate. The `text` argument can be a single word or a short sentence.)*

---

## Acknowledgments

This project is built upon the foundational work and architecture described in the following research paper:

* **DiffusionPen: Towards Controlling the Style of Handwritten Text Generation**
    * *By Nikolaidou, K., Retsinas, G., Sfikas, G., & Liwicki, M. (2024).*
