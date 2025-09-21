import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
from torchvision import transforms
from feature_extractor import ImageEncoder
from torchvision.utils import save_image
from transformers import CanineTokenizer, CanineModel

# ===================================================================
# ## Helper Classes & Functions
# ===================================================================
torch.cuda.empty_cache()
class AvgMeter:
    def __init__(self, name="Metric"): self.name = name; self.reset()
    def reset(self): self.avg, self.sum, self.count = [0] * 3
    def update(self, val, count=1): self.count += count; self.sum += val * count; self.avg = self.sum / self.count
class EMA:
    def __init__(self, beta): self.beta = beta; self.step = 0
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None: return new
        return old * self.beta + (1 - self.beta) * new
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema: self.reset_parameters(ema_model, model); self.step += 1; return
        self.update_model_average(ema_model, model); self.step += 1
    def reset_parameters(self, ema_model, model): ema_model.load_state_dict(model.state_dict())
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray((ndarr * 255).astype(np.uint8)); im.save(path)
    return im

# ===================================================================
# ## Main WordStylist Dataset Class
# ===================================================================
class WordStyleDataset(Dataset):
    def __init__(self, annotation_path, data_root, fixed_size, tokenizer, transforms, num_style_images=5):
        self.data_root = data_root; self.fixed_size = fixed_size; self.tokenizer = tokenizer
        self.transforms = transforms; self.num_style_images = num_style_images; self.data_points = []
        self.writer_image_pool = {}
        print(f"Loading annotations from: {annotation_path}")
        if not os.path.exists(annotation_path): raise FileNotFoundError(f"Annotation file not found at: {annotation_path}")
        with open(annotation_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        for line in tqdm(lines, desc="Processing annotations"):
            try:
                path, transcription = line.strip().split(None, 1); writer_id = path.split(os.sep)[-3]
                if writer_id not in self.writer_image_pool: self.writer_image_pool[writer_id] = []
                self.writer_image_pool[writer_id].append({"path": path, "transcription": transcription})
            except (ValueError, IndexError): print(f"Warning: Skipping malformed line or path: {line.strip()}"); continue
        self.writers = sorted(self.writer_image_pool.keys()); self.writer_to_id = {writer: i for i, writer in enumerate(self.writers)}
        for writer_name, items in tqdm(self.writer_image_pool.items(), desc="Creating data points"):
            if len(items) > self.num_style_images:
                writer_id_int = self.writer_to_id[writer_name]
                for item in items:
                    self.data_points.append({"path": item["path"], "transcription": item["transcription"], "writer_id_str": writer_name, "writer_id_int": writer_id_int})
        print(f"Found {len(self.data_points)} images from {len(self.writers)} writers.")
    def __len__(self): return len(self.data_points)
    def __getitem__(self, index):
        item = self.data_points[index]; anchor_path_full = os.path.join(self.data_root, item["path"])
        transcription = item["transcription"]; writer_id_int = item["writer_id_int"]; writer_id_str = item["writer_id_str"]
        anchor_img = Image.open(anchor_path_full).convert("RGB"); style_pool = self.writer_image_pool[writer_id_str]
        possible_style_paths = [os.path.join(self.data_root, p["path"]) for p in style_pool if p["path"] != item["path"]]
        style_image_paths = random.sample(possible_style_paths, self.num_style_images); style_images = []
        for p in style_image_paths: img = Image.open(p).convert("RGB"); style_images.append(self.transforms(img))
        return self.transforms(anchor_img), transcription, writer_id_int, torch.stack(style_images)

# ===================================================================
# ## Diffusion Class and Training/Sampling Loop
# ===================================================================
class Diffusion:
    def __init__(self, img_size=(64, 256), args=None): self.img_size = img_size; self.device = args.device
    def sample(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, tokenizer, style_images):
        model.eval()
        with torch.no_grad():
            text_features = tokenizer(x_text, padding="max_length", truncation=True, return_tensors="pt", max_length=95).to(self.device)
            reshaped_style_images = style_images.view(-1, 3, args.img_size[0], args.img_size[1])
            style_features = style_extractor(reshaped_style_images)
            x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(self.device)
            noise_scheduler.set_timesteps(50)
            for time in tqdm(noise_scheduler.timesteps, desc="Sampling"):
                t = (torch.ones(n) * time).long().to(self.device)
                residual = model(x, t, text_features, labels, style_extractor=style_features)
                x = noise_scheduler.step(residual, time, x).prev_sample
            latents = 1 / 0.18215 * x; image = vae.decode(latents).sample; image = (image / 2 + 0.5).clamp(0, 1)
        return image
        
def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, val_loader, args, style_extractor, noise_scheduler, tokenizer):
    for epoch in range(args.start_epoch, args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---"); model.train()
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, data in enumerate(pbar):
            images, transcr, s_id, style_images = data
            images, s_id, style_images = images.to(args.device), s_id.to(args.device), style_images.to(args.device)
            text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=95).to(args.device)
            with torch.no_grad():
                reshaped_style_images = style_images.view(-1, 3, args.img_size[0], args.img_size[1])
                style_features = style_extractor(reshaped_style_images)
            latents = vae.encode(images.to(torch.float32)).latent_dist.sample() * 0.18215; noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            predicted_noise = model(noisy_latents, timesteps, text_features, s_id, style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise); loss = loss / args.accumulation_steps; loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step(); optimizer.zero_grad(); ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item() * args.accumulation_steps)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save({'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_path, "models", "optim.pt"))
            print("Saved checkpoint.")
            try:
                val_batch = next(iter(val_loader)); _, sample_transcr, sample_labels, sample_style_images = val_batch
                sample_labels = sample_labels[:4].to(args.device); sample_style_images = sample_style_images[:4].to(args.device); sample_transcr = sample_transcr[:4]
                sampled_images = diffusion.sample(ema_model, vae, n=4, x_text=sample_transcr, labels=sample_labels, args=args, style_extractor=style_extractor, noise_scheduler=noise_scheduler, tokenizer=tokenizer, style_images=sample_style_images)
                save_images(sampled_images, os.path.join(args.save_path, 'images', f"epoch_{epoch}.jpg"), nrow=2); print("Saved sample validation image.")
            except StopIteration: print("Validation loader is empty, skipping sampling.")

# ===================================================================
# ## Main Execution Block
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train or Sample from DiffusionPen on WordStylist dataset")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to resume from (if optim.pt is not found).')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--train_annotation', type=str, help='Path to train.txt (required for training).')
    parser.add_argument('--val_annotation', type=str, help='Path to val.txt (required for training).')
    parser.add_argument('--data_root', type=str, help='Base path to dataset image folders (required for training).')
    parser.add_argument('--img_size', type=tuple, default=(64, 256))
    parser.add_argument('--save_path', type=str, default='./diffusion_models')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--style_path', type=str, required=True, help='Path to your PRE-TRAINED style encoder .pth file')
    parser.add_argument('--stable_dif_path', type=str, default='/scratch/m24csa020/m23cse027/models/stable-diffusion-v1-5')
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
    transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tokenizer = CanineTokenizer.from_pretrained("./canine-tokenizer")
    device = torch.device(args.device)

    # --- Shared Model Loading ---
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100').to(device)
    feature_extractor.load_state_dict(torch.load(args.style_path, map_location=device))
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    if args.mode == 'train':
        if not all([args.train_annotation, args.val_annotation, args.data_root]):
            parser.error("--train_annotation, --val_annotation, and --data_root are required for training.")
        train_data = WordStyleDataset(args.train_annotation, args.data_root, args.img_size, tokenizer, transform)
        val_data = WordStyleDataset(args.val_annotation, args.data_root, args.img_size, tokenizer, transform)
        args.style_classes = len(train_data.writers)
        print(f"Number of unique writers found: {args.style_classes}")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        text_encoder = CanineModel.from_pretrained("./canine-model")
        unet = UNetModel(image_size=args.img_size, in_channels=4, model_channels=args.model_channels, out_channels=4, num_res_blocks=1, attention_resolutions=(1,1), num_heads=4, num_classes=args.style_classes, context_dim=768, vocab_size=None, args=args, text_encoder=text_encoder).to(device)
        optimizer = optim.AdamW(unet.parameters(), lr=1e-4)
        mse_loss = nn.MSELoss()
        noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
        diffusion = Diffusion(img_size=args.img_size, args=args)
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

        # --- RESUME LOGIC ---
        # Set start_epoch from the command line argument as a default
        start_epoch_num = args.start_epoch
        if args.resume:
            ckpt_path = os.path.join(args.save_path, "models", "ckpt.pt")
            ema_ckpt_path = os.path.join(args.save_path, "models", "ema_ckpt.pt")
            optim_path = os.path.join(args.save_path, "models", "optim.pt")
            if os.path.exists(ckpt_path):
                print(f"Resuming model weights from checkpoint: {ckpt_path}")
                unet.load_state_dict(torch.load(ckpt_path, map_location=device))
                ema_model.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
                # If optim.pt exists, it's the most reliable source for the epoch number
                if os.path.exists(optim_path):
                    print("Resuming optimizer state and epoch count from optim.pt...")
                    optim_checkpoint = torch.load(optim_path, map_location=device)
                    optimizer.load_state_dict(optim_checkpoint['optimizer_state_dict'])
                    start_epoch_num = optim_checkpoint['epoch'] + 1 # Override with file value
                else:
                    print("Optimizer checkpoint not found. Using manually specified start epoch.")
            else:
                print("Resume flag was set, but no model checkpoint was found. Starting from scratch.")
        # Pass the determined start epoch to the training function
        args.start_epoch = start_epoch_num
        
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, val_loader, args, feature_extractor, noise_scheduler, tokenizer)

    elif args.mode == 'sample':
        # --- SAMPLING CONFIGURATION ---
        words_to_generate = ["नमस्ते", "भारत", "कंप्यूटर", "विज्ञान"]
        style_sample_folder = './style_samples/'
        output_filename = "generated_handwriting.jpg"
        # --------------------------------
        
        print("--- Starting Sampling Mode ---")
        style_image_paths = [os.path.join(style_sample_folder, f) for f in os.listdir(style_sample_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not style_image_paths:
            raise FileNotFoundError(f"No style images found in '{style_sample_folder}'. Please add 5-10 images of a single writer's handwriting.")
        style_images_list = [transform(Image.open(p).convert("RGB")) for p in style_image_paths]
        style_images = torch.stack(style_images_list).unsqueeze(0).repeat(len(words_to_generate), 1, 1, 1, 1).to(device)
        style_label = torch.tensor([0] * len(words_to_generate)).to(device)
        
        print("Loading trained model for sampling...")
        if not args.train_annotation:
             args.train_annotation = input("For sampling, please provide the path to your train_annotation.txt file to determine class count: ")
        with open(args.train_annotation, 'r', encoding='utf-8') as f:
            writers = set()
            for line in f:
                writers.add(line.strip().split(None, 1)[0].split(os.sep)[-3])
            args.style_classes = len(writers)
            
        text_encoder = CanineModel.from_pretrained("./canine-model")
        ema_unet = UNetModel(image_size=args.img_size, in_channels=4, model_channels=args.model_channels, out_channels=4, num_res_blocks=1, attention_resolutions=(1,1), num_heads=4, num_classes=args.style_classes, context_dim=768, vocab_size=None, args=args, text_encoder=text_encoder).to(device)
        
        ema_ckpt_path = os.path.join(args.save_path, "models", "ema_ckpt.pt")
        print(f"Loading EMA model from {ema_ckpt_path}")
        ema_unet.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
        
        noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
        diffusion = Diffusion(img_size=args.img_size, args=args)

        generated_images = diffusion.sample(model=ema_unet, vae=vae, n=len(words_to_generate), x_text=words_to_generate, labels=style_label, args=args, style_extractor=feature_extractor, noise_scheduler=noise_scheduler, tokenizer=tokenizer, style_images=style_images)
        
        output_path = os.path.join(args.save_path, output_filename)
        save_images(generated_images, output_path, nrow=len(words_to_generate))
        print(f"Saved generated images to {output_path}")

if __name__ == "__main__":
    main()