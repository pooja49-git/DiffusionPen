# import os
# import torch
# import numpy as np
# from PIL import Image, ImageOps
# import argparse
# from diffusers import AutoencoderKL, DDIMScheduler
# import random
# from unet import UNetModel
# from torchvision import transforms
# from feature_extractor import ImageEncoder
# from transformers import CanineTokenizer, CanineModel
# from tqdm import tqdm

# # ===================================================================
# # ## Helper function to crop whitespace
# # ===================================================================
# def crop_whitespace(image):
#     # Convert PIL image to numpy array
#     np_image = np.array(image)
#     # Get the alpha channel if it exists, otherwise use grayscale
#     if np_image.shape[2] == 4:
#         alpha = np_image[:, :, 3]
#     else:
#         # Convert to grayscale to find non-white pixels
#         gray = image.convert('L')
#         alpha = np.array(gray)

#     # Find all non-white pixels (value < 255)
#     coords = np.argwhere(alpha < 255)
#     if coords.size == 0:
#         return image # Return original if image is all white

#     # Get bounding box
#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)
    
#     return image.crop((x_min, y_min, x_max + 1, y_max + 1))

# # ===================================================================
# # ## Main Generation Function
# # ===================================================================
# def generate(args):
#     # --- Setup ---
#     device = torch.device(args.device)
#     transform = transforms.Compose([transforms.Resize((64, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     # --- Load All Models ---
#     print("Loading models...")
#     tokenizer = CanineTokenizer.from_pretrained("./canine-tokenizer")
#     text_encoder = CanineModel.from_pretrained("./canine-model")
#     vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    
#     # Load Style Encoder
#     style_encoder = ImageEncoder(model_name='mobilenetv2_100').to(device)
#     style_encoder.load_state_dict(torch.load(args.style_path, map_location=device))
#     style_encoder.eval()

#     # Load the trained U-Net Generator
#     # We need to know the number of classes the model was trained on
#     with open(args.train_annotation, 'r', encoding='utf-8') as f:
#         writers = set(line.strip().split(None, 1)[0].split(os.sep)[-3] for line in f)
#         num_classes = len(writers)

#     unet = UNetModel(image_size=(64,256), in_channels=4, model_channels=128, out_channels=4, num_res_blocks=1, attention_resolutions=(1,1), num_heads=4, num_classes=num_classes, context_dim=768, args=args, text_encoder=text_encoder).to(device)
    
#     ema_ckpt_path = os.path.join(args.save_path, "models", "ema_ckpt.pt")
#     unet.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
#     print(f"Loaded U-Net from {ema_ckpt_path}")
    
#     noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
#     # --- Prepare Inputs ---
#     words_to_generate = args.text.split()
#     print(f"Generating words: {words_to_generate}")
    
#     # Load style reference images
#     style_image_paths = [os.path.join(args.style_folder, f) for f in os.listdir(args.style_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
#     if not style_image_paths:
#         raise FileNotFoundError(f"No style images found in '{args.style_folder}'.")
    
#     style_images_list = [transform(Image.open(p).convert("RGB")) for p in style_image_paths]
#     style_images_batch = torch.stack(style_images_list).to(device)
    
#     # --- Generate each word individually ---
#     generated_words = []
#     for word in tqdm(words_to_generate, desc="Generating individual words"):
#         with torch.no_grad():
#             # Prepare inputs for this word
#             text_features = tokenizer([word], padding="max_length", truncation=True, return_tensors="pt", max_length=95).to(device)
#             style_features = style_encoder(style_images_batch)
#             style_features_avg = torch.mean(style_features, dim=0, keepdim=True) # Average the styles
            
#             x = torch.randn((1, 4, 8, 32)).to(device)
#             noise_scheduler.set_timesteps(50)

#             for time in noise_scheduler.timesteps:
#                 t = (torch.ones(1) * time).long().to(device)
#                 residual = unet(x, t, text_features, y=None, style_extractor=style_features_avg.repeat(5,1)) # Repeat avg style
#                 x = noise_scheduler.step(residual, time, x).prev_sample

#             latents = 1 / 0.18215 * x
#             image_tensor = vae.decode(latents).sample
#             image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        
#         # Convert to PIL Image
#         img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
#         generated_words.append(img)
        
#     # --- Stitch words into a paragraph ---
#     print("Stitching words into a paragraph...")
#     cropped_words = [crop_whitespace(word_img) for word_img in generated_words]
    
#     space_width = 20 # pixels between words
#     line_height = max(img.height for img in cropped_words) + 20 # max height in a line
#     max_width = 1024 # max width of the final image

#     # Create a blank canvas
#     canvas = Image.new('RGB', (max_width, line_height * len(words_to_generate)), 'white')
    
#     x_cursor, y_cursor = 0, 10
#     for img in cropped_words:
#         if x_cursor + img.width > max_width: # New line
#             x_cursor = 0
#             y_cursor += line_height
        
#         canvas.paste(img, (x_cursor, y_cursor))
#         x_cursor += img.width + space_width
        
#     # Crop the final canvas to the content
#     final_image = crop_whitespace(canvas)
    
#     output_path = os.path.join(args.save_path, args.output_filename)
#     final_image.save(output_path)
#     print(f"✅ Saved final paragraph image to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate sentences/paragraphs with a trained DiffusionPen model.")
#     parser.add_argument('--text', type=str, required=True, help='The sentence or paragraph to generate.')
#     parser.add_argument('--style_folder', type=str, default='./style_samples/', help='Folder with 5-10 images of the target handwriting style.')
#     parser.add_argument('--output_filename', type=str, default='generated_paragraph.jpg')
    
#     # Paths from your training script
#     parser.add_argument('--save_path', type=str, default='./diffusion_models')
#     parser.add_argument('--device', type=str, default='cuda:0')
#     parser.add_argument('--style_path', type=str, required=True, help='Path to your PRE-TRAINED style encoder .pth file')
#     parser.add_argument('--stable_dif_path', type=str, default='/scratch/m24csa020/m23cse027/models/stable-diffusion-v1-5')
#     parser.add_argument('--train_annotation', type=str, required=True, help='Path to the original train.txt to get the number of writers.')
    
#     # Model architecture args (must match training)
#     parser.add_argument('--model_channels', type=int, default=128)
#     parser.add_argument('--interpolation', type=bool, default=False)
#     parser.add_argument('--mix_rate', type=float, default=None)
#     args = parser.parse_args()
    
#     generate(args)

import os
import torch
import numpy as np
from PIL import Image
import argparse
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
from torchvision import transforms
from feature_extractor import ImageEncoder
from transformers import CanineTokenizer, CanineModel
from tqdm import tqdm

# ===================================================================
# ## Helper function to crop whitespace (More Robust Version)
# ===================================================================
def crop_whitespace(image):
    # Convert to grayscale and get the bounding box of non-white pixels.
    # The getbbox() method is a simple and effective way to find the content.
    bbox = image.convert("L").getbbox()
    if bbox:
        return image.crop(bbox)
    return image # Return original image if it's all white

# ===================================================================
# ## Main Generation Function
# ===================================================================
def generate(args):
    # --- Setup ---
    device = torch.device(args.device)
    transform = transforms.Compose([transforms.Resize((64, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # --- Load All Models ---
    print("Loading models...")
    tokenizer = CanineTokenizer.from_pretrained("./canine-tokenizer")
    text_encoder = CanineModel.from_pretrained("./canine-model")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    
    style_encoder = ImageEncoder(model_name='mobilenetv2_100').to(device)
    style_encoder.load_state_dict(torch.load(args.style_path, map_location=device))
    style_encoder.eval()

    with open(args.train_annotation, 'r', encoding='utf-8') as f:
        writers = set(line.strip().split(None, 1)[0].split(os.sep)[-3] for line in f)
        num_classes = len(writers)

    unet = UNetModel(image_size=(64,256), in_channels=4, model_channels=128, out_channels=4, num_res_blocks=1, attention_resolutions=(1,1), num_heads=4, num_classes=num_classes, context_dim=768, args=args, text_encoder=text_encoder).to(device)
    
    ema_ckpt_path = os.path.join(args.save_path, "models", "ema_ckpt.pt")
    unet.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
    print(f"Loaded U-Net from {ema_ckpt_path}")
    
    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # --- Prepare Inputs ---
    words_to_generate = args.text.split()
    print(f"Generating words: {words_to_generate}")
    
    style_image_paths = [os.path.join(args.style_folder, f) for f in os.listdir(args.style_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not style_image_paths:
        raise FileNotFoundError(f"No style images found in '{args.style_folder}'.")
    
    style_images_list = [transform(Image.open(p).convert("RGB")) for p in style_image_paths]
    style_images_batch = torch.stack(style_images_list).to(device)
    
    # --- Generate each word individually ---
    generated_words = []
    for word in tqdm(words_to_generate, desc="Generating individual words"):
        with torch.no_grad():
            text_features = tokenizer([word], padding="max_length", truncation=True, return_tensors="pt", max_length=95).to(device)
            style_features = style_encoder(style_images_batch)
            style_features_avg = torch.mean(style_features, dim=0, keepdim=True)
            
            x = torch.randn((1, 4, 8, 32)).to(device)
            noise_scheduler.set_timesteps(50)

            for time in noise_scheduler.timesteps:
                t = (torch.ones(1) * time).long().to(device)
                residual = unet(x, t, text_features, y=None, style_extractor=style_features_avg.repeat(5,1))
                x = noise_scheduler.step(residual, time, x).prev_sample

            latents = 1 / 0.18215 * x
            image_tensor = vae.decode(latents).sample
            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        
        img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        generated_words.append(img)
        
    # --- Stitch words into a paragraph ---
    print("Stitching words into a paragraph...")
    cropped_words = [crop_whitespace(word_img) for word_img in generated_words]
    
    space_width = 20
    line_height = max(img.height for img in cropped_words if img) + 20
    max_width = 1024

    canvas = Image.new('RGB', (max_width, line_height * len(words_to_generate)), 'white')
    
    x_cursor, y_cursor = 10, 10
    for img in cropped_words:
        if img is None: continue
        if x_cursor + img.width > max_width:
            x_cursor = 10
            y_cursor += line_height
        
        canvas.paste(img, (x_cursor, y_cursor))
        x_cursor += img.width + space_width
        
    final_image = crop_whitespace(canvas)
    
    output_path = os.path.join(args.save_path, args.output_filename)
    final_image.save(output_path)
    print(f"✅ Saved final paragraph image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentences/paragraphs with a trained DiffusionPen model.")
    parser.add_argument('--text', type=str, required=True, help='The sentence or paragraph to generate.')
    parser.add_argument('--style_folder', type=str, default='./style_samples/', help='Folder with 5-10 images of the target handwriting style.')
    parser.add_argument('--output_filename', type=str, default='generated_paragraph.jpg')
    
    parser.add_argument('--save_path', type=str, default='./diffusion_models')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--style_path', type=str, required=True, help='Path to your PRE-TRAINED style encoder .pth file')
    parser.add_argument('--stable_dif_path', type=str, default='/scratch/m24csa020/m23cse027/models/stable-diffusion-v1-5')
    parser.add_argument('--train_annotation', type=str, required=True, help='Path to the original train.txt to get the number of writers.')
    
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    args = parser.parse_args()
    
    generate(args)