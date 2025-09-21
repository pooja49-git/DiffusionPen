# # import torch
# # import torch.nn as nn
# # import torchvision.models as models
# # from torchvision import transforms
# # from torch.utils.data import DataLoader, Dataset, random_split
# # import numpy as np
# # from PIL import Image, ImageOps
# # from os.path import isfile
# # from skimage import io
# # from torchvision.utils import save_image
# # from skimage.transform import resize
# # import os
# # import argparse
# # import torch.optim as optim
# # from tqdm import tqdm
# # from utils.iam_dataset import IAMDataset
# # from utils.auxilary_functions import affine_transformation
# # from feature_extractor import ImageEncoder
# # import timm
# # import cv2
# # import time
# # import json
# # import random


# # class AvgMeter:
# #     def __init__(self, name="Metric"):
# #         self.name = name
# #         self.reset()

# #     def reset(self):
# #         self.avg, self.sum, self.count = [0] * 3

# #     def update(self, val, count=1):
# #         self.count += count
# #         self.sum += val * count
# #         self.avg = self.sum / self.count

# #     def __repr__(self):
# #         text = f"{self.name}: {self.avg:.4f}"
# #         return text


# # class WordStyleDataset(Dataset):
# #     '''
# #     Dataset class for word stylistics.
# #     This version automatically scans a directory structure like:
# #     basefolder/writer_id/session_id/image.jpg
# #     '''
# #     def __init__(self, 
# #         basefolder: str,
# #         subset: str = 'all',
# #         segmentation_level: str = 'word',
# #         fixed_size: tuple =(128, None),
# #         transforms: list = None,
# #         character_classes: list = None,
# #         ):
        
# #         self.basefolder = basefolder
# #         self.subset = subset
# #         self.segmentation_level = segmentation_level
# #         self.fixed_size = fixed_size
# #         self.transforms = transforms
# #         self.data_info = []

# #         # --- MODIFIED SECTION: Scan directory instead of reading a file ---
# #         print(f"Scanning dataset folder: {self.basefolder}")
# #         if not os.path.isdir(self.basefolder):
# #             raise FileNotFoundError(f"The provided dataset path does not exist: {self.basefolder}")

# #         for writer_id in tqdm(os.listdir(self.basefolder), desc="Loading writers"):
# #             writer_path = os.path.join(self.basefolder, writer_id)
# #             if os.path.isdir(writer_path):
# #                 for session_folder in os.listdir(writer_path):
# #                     session_path = os.path.join(writer_path, session_folder)
# #                     if os.path.isdir(session_path):
# #                         for image_name in os.listdir(session_path):
# #                             if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
# #                                 image_path = os.path.join(session_path, image_name)
# #                                 # NOTE: Transcription is a placeholder as it's not in the folder structure.
# #                                 transcription = "" 
# #                                 self.data_info.append([image_path, writer_id, transcription])
        
# #         print(f"Found {len(self.data_info)} images from {len(os.listdir(self.basefolder))} writers.")
# #         # --- END OF MODIFIED SECTION ---
        
# #     def __len__(self):
# #         return len(self.data_info)
    
# #     def __getitem__(self, index):
        
# #         img_path, wid, transcr = self.data_info[index]
# #         img = Image.open(img_path).convert('RGB')

# #         # Pick a positive sample (different image, same writer)
# #         # and a negative sample (different image, different writer)
# #         positive_samples = [p for p in self.data_info if p[1] == wid]
# #         negative_samples = [n for n in self.data_info if n[1] != wid]
        
# #         positive_path = random.choice(positive_samples)[0]
# #         while positive_path == img_path: # Ensure we don't pick the same image
# #              positive_path = random.choice(positive_samples)[0]

# #         negative_path = random.choice(negative_samples)[0]

# #         img_pos = Image.open(positive_path).convert('RGB')
# #         img_neg = Image.open(negative_path).convert('RGB')
        
# #         # --- Image processing and augmentation ---
# #         fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        
# #         def process_image(image_pil):
# #             if self.subset == 'train':
# #                 # Apply some augmentation for training
# #                 nwidth = int(np.random.uniform(0.75, 1.25) * image_pil.width)
# #                 nheight = int((np.random.uniform(0.9, 1.1) * image_pil.height / image_pil.width) * nwidth)
# #             else:
# #                 nheight, nwidth = image_pil.height, image_pil.width
            
# #             nheight = max(4, min(fheight - 16, nheight))
# #             nwidth = max(8, min(fwidth - 32, nwidth))
            
# #             resized_img = image_resize_PIL(image_pil, height=nheight, width=nwidth)
# #             centered_img = centered_PIL(resized_img, (fheight, fwidth), border_value=255.0)
# #             return centered_img

# #         img = process_image(img)
# #         img_pos = process_image(img_pos)
# #         img_neg = process_image(img_neg)
        
# #         if self.transforms is not None:
# #             img = self.transforms(img)
# #             img_pos = self.transforms(img_pos)
# #             img_neg = self.transforms(img_neg)
        
# #         return img, transcr, wid, img_pos, img_neg, img_path

# #     def collate_fn(self, batch):
# #         img, transcr, wid, positive, negative, img_path = zip(*batch)
# #         images_batch = torch.stack(img)
# #         images_pos = torch.stack(positive)
# #         images_neg = torch.stack(negative)
# #         return images_batch, transcr, wid, images_pos, images_neg, img_path


# # def image_resize_PIL(img, height=None, width=None):
# #     if height is None and width is None:
# #         return img

# #     original_width, original_height = img.size

# #     if height is not None and width is None:
# #         aspect_ratio = original_width / original_height
# #         new_width = int(height * aspect_ratio)
# #         new_height = height
# #     elif width is not None and height is None:
# #         aspect_ratio = original_height / original_width
# #         new_height = int(width * aspect_ratio)
# #         new_width = width
# #     else:
# #         new_width = width
# #         new_height = height

# #     resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
# #     return resized_img


# # def centered_PIL(word_img, tsize, centering=(.5, .5), border_value=None):
# #     height, width = tsize
# #     res = Image.new('RGB', (width, height), color=(255, 255, 255))
    
# #     img_w, img_h = word_img.size
    
# #     # Calculate padding
# #     pad_left = (width - img_w) // 2
# #     pad_top = (height - img_h) // 2
    
# #     res.paste(word_img, (pad_left, pad_top))
# #     return res

# # # --- Other classes and functions from the original script remain the same ---
# # # (WordLineDataset, IAMDataset_style, Mixed_Encoder, performance, training loops, etc.)
# # # They are omitted here for brevity but are assumed to be present in the final file.
# # # The important part is the change in WordStyleDataset and its usage in main().
# # class WordLineDataset(Dataset):
# #     def __init__(self, 
# #         basefolder: str = 'datasets/',
# #         subset: str = 'all',
# #         segmentation_level: str = 'line',
# #         fixed_size: tuple =(128, None),
# #         transforms: list = None,
# #         character_classes: list = None,
# #         ):
# #         pass # This class is now a placeholder and not used directly

# # class IAMDataset_style(WordLineDataset):
# #      def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
# #         pass # This class is now a placeholder and not used directly

# # class Mixed_Encoder(nn.Module):
# #     def __init__(
# #         self, model_name='resnet50', num_classes=339, pretrained=True, trainable=True
# #     ):
# #         super().__init__()
# #         self.model = timm.create_model(
# #             model_name, pretrained, num_classes=0, global_pool=""
# #         )
# #         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
# #         num_features = self.model.num_features
# #         self.classifier = nn.Linear(num_features, num_classes)
# #         for p in self.model.parameters():
# #             p.requires_grad = trainable
            
# #     def forward(self, x):
# #         features = self.model(x)
# #         pooled_features = self.global_pool(features).flatten(1)
# #         logits = self.classifier(pooled_features)
# #         return logits, pooled_features  

# # def performance(pred, label):
# #     loss = nn.CrossEntropyLoss()
# #     loss = loss(pred, label)
# #     return loss 

# # def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
# #     model.train()
# #     loss_meter = AvgMeter()
# #     pbar = tqdm(train_loader, desc="Training Triplet")
# #     for data in pbar:
# #         anchor, _, _, positive, negative, _ = data
# #         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
# #         _, anchor_out = model(anchor)
# #         _, positive_out = model(positive)
# #         _, negative_out = model(negative)
        
# #         loss = criterion(anchor_out, positive_out, negative_out)
        
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
        
# #         loss_meter.update(loss.item(), anchor.size(0))
# #         pbar.set_postfix(triplet_loss=loss_meter.avg)
        
# #     print(f"Training Loss: {loss_meter.avg:.4f}")
# #     return loss_meter.avg

# # def val_epoch_triplet(val_loader, model, criterion, device, args):
# #     model.eval()
# #     loss_meter = AvgMeter()
# #     pbar = tqdm(val_loader, desc="Validating Triplet")
# #     with torch.no_grad():
# #         for data in pbar:
# #             anchor, _, _, positive, negative, _ = data
# #             anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

# #             _, anchor_out = model(anchor)
# #             _, positive_out = model(positive)
# #             _, negative_out = model(negative)
            
# #             loss = criterion(anchor_out, positive_out, negative_out)
# #             loss_meter.update(loss.item(), anchor.size(0))
# #             pbar.set_postfix(triplet_loss=loss_meter.avg)

# #     print(f"Validation Loss: {loss_meter.avg:.4f}")
# #     return loss_meter.avg

# # def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
# #     best_loss = float('inf')
# #     for epoch_i in range(args.epochs):
# #         print(f"--- Epoch {epoch_i+1}/{args.epochs} ---")
# #         train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
# #         val_loss = val_epoch_triplet(val_loader, model, criterion, device, args)
        
# #         if val_loss < best_loss:
# #             best_loss = val_loss
# #             if not os.path.exists(args.save_path):
# #                 os.makedirs(args.save_path)
# #             save_file = os.path.join(args.save_path, f'triplet_{args.dataset}_{args.model}.pth')
# #             torch.save(model.state_dict(), save_file)
# #             print(f"Saved Best Model! (Loss: {best_loss:.4f})")
        
# #         scheduler.step(val_loss)


# # def main():
# #     '''Main function'''
# #     parser = argparse.ArgumentParser(description='Train Style Encoder')
# #     parser.add_argument('--model', type=str, default='mobilenetv2_100', help='type of cnn to use')
# #     parser.add_argument('--dataset', type=str, default='wordstylist', help='dataset name')
# #     parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
# #     parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
# #     parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training')
# #     parser.add_argument('--save_path', type=str, default='/csehome/m24csa020/DiffusionPen/DiffusionPen-main/style_models', help='path to save models')
# #     # The mode is now implicitly 'triplet' based on the dataset structure
# #     parser.add_argument('--mode', type=str, default='triplet', help='triplet, mixed, or classification')

# #     args = parser.parse_args()
# #     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")

# #     # --- UPDATED DATA LOADING SECTION ---
# #     data_transform = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #     ])

# #     # Path to the root of your training data
# #     dataset_base_path = '/scratch/m24csa020/m23cse027/wordstylist/HindiSeg/train/'

# #     full_dataset = WordStyleDataset(
# #         basefolder=dataset_base_path,
# #         fixed_size=(64, 256),
# #         transforms=data_transform,
# #         subset='train'
# #     )

# #     if len(full_dataset) == 0:
# #         print("No images found in the dataset path. Please check the path and folder structure.")
# #         return

# #     validation_size = int(0.1 * len(full_dataset))
# #     train_size = len(full_dataset) - validation_size
# #     train_data, val_data = random_split(full_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

# #     print(f'Total dataset size: {len(full_dataset)}')
# #     print(f'Training set size: {len(train_data)}')
# #     print(f'Validation set size: {len(val_data)}')

# #     # Use the collate_fn from the original dataset instance for both splits
# #     collate_function = full_dataset.collate_fn

# #     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_function)
# #     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_function)
    
# #     # --- MODEL AND TRAINING SETUP ---
# #     # The number of classes for the classification head is not strictly needed for triplet loss,
# #     # but we can set it to the number of writers found.
# #     num_writers = len(os.listdir(dataset_base_path))

# #     model = Mixed_Encoder(model_name=args.model, num_classes=num_writers, pretrained=True, trainable=True)
# #     model = model.to(device)
# #     print(f'Number of model parameters: {sum(p.data.nelement() for p in model.parameters())}')
    
# #     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1, verbose=True)
# #     criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# #     if args.mode == 'triplet':
# #         print("Starting training in 'triplet' mode...")
# #         train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args)
# #     else:
# #         print(f"Warning: The current dataset loader is optimized for 'triplet' mode. Running in '{args.mode}' mode may not work as expected.")
    
# #     print('Finished training.')


# # if __name__ == '__main__':
# #     main()

# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset, random_split
# import numpy as np
# from PIL import Image, ImageOps
# import os
# import argparse
# import torch.optim as optim
# from tqdm import tqdm
# import timm
# import random

# # ===================================================================
# # ## Helper Class for Averaging Metrics
# # ===================================================================
# class AvgMeter:
#     def __init__(self, name="Metric"):
#         self.name = name
#         self.reset()

#     def reset(self):
#         self.avg, self.sum, self.count = [0] * 3

#     def update(self, val, count=1):
#         self.count += count
#         self.sum += val * count
#         self.avg = self.sum / self.count

#     def __repr__(self):
#         text = f"{self.name}: {self.avg:.4f}"
#         return text

# # ===================================================================
# # ## Custom Dataset for Your WordStylist Data
# # ===================================================================
# class WordStyleDataset(Dataset):
#     '''
#     Dataset class for word stylistics.
#     This version automatically scans a directory structure like:
#     basefolder/writer_id/session_id/image.jpg
#     '''
#     def __init__(self, 
#         basefolder: str,
#         subset: str = 'train',
#         fixed_size: tuple =(64, 256),
#         transforms: list = None
#         ):
        
#         self.basefolder = basefolder
#         self.subset = subset
#         self.fixed_size = fixed_size
#         self.transforms = transforms
#         self.data_info = []

#         print(f"Scanning dataset folder: {self.basefolder}")
#         if not os.path.isdir(self.basefolder):
#             raise FileNotFoundError(f"The provided dataset path does not exist: {self.basefolder}")

#         # Scan directory to build the list of images and writer IDs
#         for writer_id in tqdm(os.listdir(self.basefolder), desc="Loading writers"):
#             writer_path = os.path.join(self.basefolder, writer_id)
#             if os.path.isdir(writer_path):
#                 image_paths_for_writer = []
#                 for session_folder in os.listdir(writer_path):
#                     session_path = os.path.join(writer_path, session_folder)
#                     if os.path.isdir(session_path):
#                         for image_name in os.listdir(session_path):
#                             if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                                 image_paths_for_writer.append(os.path.join(session_path, image_name))
                
#                 # Only add writer if they have more than one image (required for triplets)
#                 if len(image_paths_for_writer) > 1:
#                     for image_path in image_paths_for_writer:
#                         # NOTE: Transcription is a placeholder as it's not in the folder structure.
#                         transcription = "" 
#                         self.data_info.append([image_path, writer_id, transcription])
        
#         print(f"Found {len(self.data_info)} usable images.")
        
#     def __len__(self):
#         return len(self.data_info)
    
#     def __getitem__(self, index):
#         img_path, wid, transcr = self.data_info[index]
#         img = Image.open(img_path).convert('RGB')

#         # Pick a positive sample (different image, same writer)
#         # and a negative sample (different image, different writer)
#         positive_samples = [p for p in self.data_info if p[1] == wid and p[0] != img_path]
#         negative_samples = [n for n in self.data_info if n[1] != wid]
        
#         positive_path = random.choice(positive_samples)[0]
#         negative_path = random.choice(negative_samples)[0]

#         img_pos = Image.open(positive_path).convert('RGB')
#         img_neg = Image.open(negative_path).convert('RGB')
        
#         # --- Image processing and augmentation ---
#         fheight, fwidth = self.fixed_size
        
#         def process_image(image_pil):
#             if self.subset == 'train':
#                 # Apply some augmentation for training
#                 nwidth = int(np.random.uniform(0.8, 1.2) * image_pil.width)
#                 nheight = int((np.random.uniform(0.8, 1.2) * image_pil.height / image_pil.width) * nwidth)
#             else:
#                 nheight, nwidth = image_pil.height, image_pil.width
            
#             nheight = max(8, min(fheight - 8, nheight))
#             nwidth = max(8, min(fwidth - 8, nwidth))
            
#             resized_img = image_resize_PIL(image_pil, height=nheight, width=nwidth)
#             centered_img = centered_PIL(resized_img, (fheight, fwidth))
#             return centered_img

#         img = process_image(img)
#         img_pos = process_image(img_pos)
#         img_neg = process_image(img_neg)
        
#         if self.transforms is not None:
#             img = self.transforms(img)
#             img_pos = self.transforms(img_pos)
#             img_neg = self.transforms(img_neg)
        
#         return img, transcr, wid, img_pos, img_neg, img_path

#     @staticmethod
#     def collate_fn(batch):
#         img, transcr, wid, positive, negative, img_path = zip(*batch)
#         images_batch = torch.stack(img)
#         images_pos = torch.stack(positive)
#         images_neg = torch.stack(negative)
#         return images_batch, transcr, wid, images_pos, images_neg, img_path

# # ===================================================================
# # ## Image Processing Helper Functions
# # ===================================================================
# def image_resize_PIL(img, height=None, width=None):
#     if height is None and width is None:
#         return img

#     original_width, original_height = img.size

#     if height is not None and width is None:
#         aspect_ratio = original_width / original_height
#         new_width = int(height * aspect_ratio)
#         new_height = height
#     elif width is not None and height is None:
#         aspect_ratio = original_height / original_width
#         new_height = int(width * aspect_ratio)
#         new_width = width
#     else:
#         new_width = width
#         new_height = height

#     return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# def centered_PIL(word_img, tsize):
#     height, width = tsize
#     # Create a new white background image
#     res = Image.new('RGB', (width, height), color=(255, 255, 255))
    
#     img_w, img_h = word_img.size
    
#     # Calculate top-left corner for pasting the image to center it
#     pad_left = (width - img_w) // 2
#     pad_top = (height - img_h) // 2
    
#     res.paste(word_img, (pad_left, pad_top))
#     return res

# # ===================================================================
# # ## Model Architecture
# # ===================================================================
# class Mixed_Encoder(nn.Module):
#     def __init__(self, model_name='resnet50', num_classes=339, pretrained=True, trainable=True):
#         super().__init__()
#         # Create a feature extractor from timm
#         self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        
#         # Add a global average pooling layer
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Get the number of features from the model
#         num_features = self.model.num_features
        
#         # Create the classifier head
#         self.classifier = nn.Linear(num_features, num_classes)

#         # Set model parameters to be trainable or frozen
#         for p in self.model.parameters():
#             p.requires_grad = trainable
            
#     def forward(self, x):
#         features = self.model(x)
#         # Get fixed-size feature vector
#         pooled_features = self.global_pool(features).flatten(1)
#         # Get classification logits
#         logits = self.classifier(pooled_features)
#         # Return both for mixed training, but we only need pooled_features for triplet loss
#         return logits, pooled_features

# # ===================================================================
# # ## Training and Validation Loops
# # ===================================================================
# def train_epoch_triplet(train_loader, model, criterion, optimizer, device):
#     model.train()
#     loss_meter = AvgMeter()
#     pbar = tqdm(train_loader, desc="Training")
    
#     for data in pbar:
#         anchor, _, _, positive, negative, _ = data
#         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
#         # Get embeddings from the model
#         _, anchor_out = model(anchor)
#         _, positive_out = model(positive)
#         _, negative_out = model(negative)
        
#         # Calculate triplet loss
#         loss = criterion(anchor_out, positive_out, negative_out)
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         loss_meter.update(loss.item(), anchor.size(0))
#         pbar.set_postfix(triplet_loss=loss_meter.avg)
        
#     print(f"Avg Training Loss: {loss_meter.avg:.4f}")
#     return loss_meter.avg

# def val_epoch_triplet(val_loader, model, criterion, device):
#     model.eval()
#     loss_meter = AvgMeter()
#     pbar = tqdm(val_loader, desc="Validating")
    
#     with torch.no_grad():
#         for data in pbar:
#             anchor, _, _, positive, negative, _ = data
#             anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

#             _, anchor_out = model(anchor)
#             _, positive_out = model(positive)
#             _, negative_out = model(negative)
            
#             loss = criterion(anchor_out, positive_out, negative_out)
#             loss_meter.update(loss.item(), anchor.size(0))
#             pbar.set_postfix(val_loss=loss_meter.avg)

#     print(f"Avg Validation Loss: {loss_meter.avg:.4f}")
#     return loss_meter.avg

# def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
#     best_loss = float('inf')
#     for epoch_i in range(args.epochs):
#         print(f"\n--- Epoch {epoch_i + 1}/{args.epochs} ---")
        
#         train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device)
#         val_loss = val_epoch_triplet(val_loader, model, criterion, device)
        
#         # Learning rate scheduler step
#         scheduler.step(val_loss)
        
#         # Save the best model
#         if val_loss < best_loss:
#             best_loss = val_loss
#             os.makedirs(args.save_path, exist_ok=True)
#             save_file = os.path.join(args.save_path, f'triplet_{args.dataset}_{args.model}_best.pth')
#             torch.save(model.state_dict(), save_file)
#             print(f"✅ Saved Best Model! (Validation Loss: {best_loss:.4f})")

# # ===================================================================
# # ## Main Execution Block
# # ===================================================================
# def main():
#     parser = argparse.ArgumentParser(description='Train Style Encoder with Triplet Loss')
#     parser.add_argument('--model', type=str, default='mobilenetv2_100', help='CNN model from timm library')
#     parser.add_argument('--dataset', type=str, default='wordstylist', help='A name for your dataset')
#     parser.add_argument('--data_path', type=str, default='/scratch/m24csa020/m23cse027/wordstylist/HindiSeg/train/', help='Path to the root of your training data')
#     parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
#     parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., cuda:0 or cpu)')
#     parser.add_argument('--save_path', type=str, default='/csehome/m24csa020/DiffusionPen/DiffusionPen-main/style_models', help='Path to save trained models')
    
#     args = parser.parse_args()
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     full_dataset = WordStyleDataset(
#         basefolder=args.data_path,
#         fixed_size=(64, 256),
#         transforms=data_transform,
#         subset='train'
#     )

#     if len(full_dataset) == 0:
#         print("Error: No images found in the dataset path. Please check the --data_path argument and your folder structure.")
#         return

#     # Split dataset into training and validation
#     validation_split = 0.1
#     validation_size = int(validation_split * len(full_dataset))
#     train_size = len(full_dataset) - validation_size
#     train_data, val_data = random_split(full_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

#     print(f'Total dataset size: {len(full_dataset)}')
#     print(f'Training set size: {len(train_data)}')
#     print(f'Validation set size: {len(val_data)}')

#     collate_function = WordStyleDataset.collate_fn
#     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_function, pin_memory=True)
#     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_function, pin_memory=True)
    
#     # Model Setup
#     writer_folders = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
#     num_writers = len(writer_folders)
#     print(f"Found {num_writers} writer styles in the dataset.")

#     model = Mixed_Encoder(model_name=args.model, num_classes=num_writers, pretrained=True, trainable=True)
#     model = model.to(device)
#     print(f'Number of model parameters: {sum(p.data.nelement() for p in model.parameters())}')
    
#     # Optimizer, Scheduler, and Loss Function
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1, verbose=True)
#     criterion = nn.TripletMarginLoss(margin=1.0, p=2)

#     print("Starting training...")
#     train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args)
    
#     print('Finished training.')

# if __name__ == '__main__':
#     main()

# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# from PIL import Image, ImageOps
# import os
# import argparse
# import torch.optim as optim
# from tqdm import tqdm
# import timm
# import random

# # ===================================================================
# # ## Helper Class & Functions
# # ===================================================================
# class AvgMeter:
#     def __init__(self, name="Metric"):
#         self.name = name
#         self.reset()
#     def reset(self):
#         self.avg, self.sum, self.count = [0] * 3
#     def update(self, val, count=1):
#         self.count += count
#         self.sum += val * count
#         self.avg = self.sum / self.count

# def image_resize_PIL(img, height=None, width=None):
#     if height is None and width is None: return img
#     original_width, original_height = img.size
#     if height is not None and width is None:
#         aspect_ratio = original_width / original_height
#         new_width = int(height * aspect_ratio)
#         new_height = height
#     else:
#         new_width = width
#         new_height = height
#     return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# def centered_PIL(word_img, tsize):
#     height, width = tsize
#     res = Image.new('RGB', (width, height), color=(255, 255, 255))
#     img_w, img_h = word_img.size
#     pad_left = (width - img_w) // 2
#     pad_top = (height - img_h) // 2
#     res.paste(word_img, (pad_left, pad_top))
#     return res

# # ===================================================================
# # ## Style Encoder Dataset Class (Reads Annotation File)
# # ===================================================================
# class StyleEncoderDataset(Dataset):
#     def __init__(self, annotation_path, data_root, fixed_size, transforms, subset='train'):
#         self.data_root = data_root 
#         self.fixed_size = fixed_size
#         self.transforms = transforms
#         self.subset = subset
        
#         writer_pool = {}
#         print(f"Loading annotations for Style Encoder from: {annotation_path}")
#         if not os.path.exists(annotation_path):
#             raise FileNotFoundError(f"Annotation file not found at: {annotation_path}")

#         with open(annotation_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()

#         for line in tqdm(lines, desc="Processing annotations"):
#             try:
#                 path, _ = line.strip().split(None, 1)
#                 writer_id = path.split(os.sep)[-3]
#                 if writer_id not in writer_pool:
#                     writer_pool[writer_id] = []
#                 writer_pool[writer_id].append(path)
#             except (ValueError, IndexError):
#                 print(f"Warning: Skipping malformed line or path: {line.strip()}")
#                 continue
        
#         self.data = []
#         for writer_id, paths in writer_pool.items():
#             if len(paths) > 1:
#                 for path in paths:
#                     self.data.append({"path": path, "writer_id": writer_id, "pool": paths})
#         print(f"Found {len(self.data)} images for style encoder training.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]
        
#         anchor_path_relative = item["path"]
#         anchor_path_full = os.path.join(self.data_root, anchor_path_relative)
#         writer_id = item["writer_id"]
        
#         positive_pool = [os.path.join(self.data_root, p) for p in item["pool"] if p != anchor_path_relative]
#         positive_path = random.choice(positive_pool)
        
#         negative_item = random.choice(self.data)
#         while negative_item["writer_id"] == writer_id:
#             negative_item = random.choice(self.data)
#         negative_path = os.path.join(self.data_root, negative_item["path"])

#         # Load images
#         anchor_img = Image.open(anchor_path_full).convert('RGB')
#         positive_img = Image.open(positive_path).convert('RGB')
#         negative_img = Image.open(negative_path).convert('RGB')

#         # Process images
#         fheight, fwidth = self.fixed_size
#         def process_image(img):
#             if self.subset == 'train':
#                 nwidth = int(np.random.uniform(0.8, 1.2) * img.width)
#                 nheight = int((np.random.uniform(0.8, 1.2) * img.height / img.width) * nwidth)
#             else:
#                 nheight, nwidth = img.height, img.width
            
#             nheight = max(8, min(fheight - 8, nheight))
#             nwidth = max(8, min(fwidth - 8, nwidth))
#             resized = image_resize_PIL(img, height=nheight, width=nwidth)
#             return centered_PIL(resized, (fheight, fwidth))
        
#         anchor_img, positive_img, negative_img = map(process_image, [anchor_img, positive_img, negative_img])

#         if self.transforms:
#             anchor_img, positive_img, negative_img = map(self.transforms, [anchor_img, positive_img, negative_img])

#         return anchor_img, positive_img, negative_img

#     @staticmethod
#     def collate_fn(batch):
#         anchors, positives, negatives = zip(*batch)
#         return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# # ===================================================================
# # ## Model Architecture
# # ===================================================================
# class ImageEncoder(nn.Module):
#     def __init__(self, model_name='mobilenetv2_100', pretrained=True):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
#         for p in self.model.parameters():
#             p.requires_grad = True

#     def forward(self, x):
#         return self.model(x)

# # ===================================================================
# # ## Training and Validation Loops
# # ===================================================================
# def train_epoch(loader, model, criterion, optimizer, device):
#     model.train()
#     loss_meter = AvgMeter()
#     pbar = tqdm(loader, desc="Training")
#     for anchor, positive, negative in pbar:
#         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
#         anchor_out = model(anchor)
#         positive_out = model(positive)
#         negative_out = model(negative)
        
#         loss = criterion(anchor_out, positive_out, negative_out)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         loss_meter.update(loss.item(), anchor.size(0))
#         pbar.set_postfix(loss=loss_meter.avg)
#     return loss_meter.avg

# def val_epoch(loader, model, criterion, device):
#     model.eval()
#     loss_meter = AvgMeter()
#     pbar = tqdm(loader, desc="Validating")
#     with torch.no_grad():
#         for anchor, positive, negative in pbar:
#             anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
#             anchor_out = model(anchor)
#             positive_out = model(positive)
#             negative_out = model(negative)
#             loss = criterion(anchor_out, positive_out, negative_out)
#             loss_meter.update(loss.item(), anchor.size(0))
#             pbar.set_postfix(loss=loss_meter.avg)
#     return loss_meter.avg

# # ===================================================================
# # ## Main Execution Block
# # ===================================================================
# def main():
#     parser = argparse.ArgumentParser(description='Train Style Encoder with Triplet Loss')
#     parser.add_argument('--model', type=str, default='mobilenetv2_100', help='CNN model from timm library')
#     parser.add_argument('--train_annotation', type=str, required=True, help='Path to your train.txt annotation file.')
#     parser.add_argument('--val_annotation', type=str, required=True, help='Path to your val.txt annotation file.')
#     # <-- FIXED: ADDED THIS LINE
#     parser.add_argument('--data_root', type=str, required=True, help='The base path to your dataset image folders (e.g., /scratch/.../wordstylist/).')
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--device', type=str, default='cuda:0')
#     parser.add_argument('--save_path', type=str, default='./style_models', help='Path to save trained models')
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     os.makedirs(args.save_path, exist_ok=True)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     train_data = StyleEncoderDataset(args.train_annotation, args.data_root, fixed_size=(64, 256), transforms=transform, subset='train')
#     val_data = StyleEncoderDataset(args.val_annotation, args.data_root, fixed_size=(64, 256), transforms=transform, subset='val')
    
#     if len(train_data) == 0 or len(val_data) == 0:
#         print("Error: One or both datasets are empty. Please check your annotation files and paths.")
#         return

#     print(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")
    
#     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=StyleEncoderDataset.collate_fn)
#     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=StyleEncoderDataset.collate_fn)

#     model = ImageEncoder(model_name=args.model).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
#     criterion = nn.TripletMarginLoss(margin=0.2)

#     best_loss = float('inf')
#     for epoch in range(args.epochs):
#         print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
#         train_loss = train_epoch(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
#         val_loss = val_epoch(loader=val_loader, model=model, criterion=criterion, device=device)
#         scheduler.step(val_loss)
        
#         if val_loss < best_loss:
#             best_loss = val_loss
#             save_file = os.path.join(args.save_path, f'wordstylist_{args.model}_best.pth')
#             torch.save(model.state_dict(), save_file)
#             print(f"✅ Saved Best Model! (Validation Loss: {best_loss:.4f})")
            
#     print("Finished training style encoder.")

# if __name__ == '__main__':
#     main()

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageOps
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
import timm
import random

# ===================================================================
# ## Helper Class & Functions
# ===================================================================
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

def image_resize_PIL(img, height=None, width=None):
    if height is None and width is None: return img
    original_width, original_height = img.size
    if height is not None and width is None:
        aspect_ratio = original_width / original_height
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = height
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def centered_PIL(word_img, tsize):
    height, width = tsize
    res = Image.new('RGB', (width, height), color=(255, 255, 255))
    img_w, img_h = word_img.size
    pad_left = (width - img_w) // 2
    pad_top = (height - img_h) // 2
    res.paste(word_img, (pad_left, pad_top))
    return res

# ===================================================================
# ## Style Encoder Dataset Class (Reads Annotation File)
# ===================================================================
class StyleEncoderDataset(Dataset):
    def __init__(self, annotation_path, data_root, fixed_size, transforms, subset='train'):
        self.data_root = data_root 
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.subset = subset
        
        writer_pool = {}
        print(f"Loading annotations for Style Encoder from: {annotation_path}")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found at: {annotation_path}")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing annotations"):
            try:
                path, _ = line.strip().split(None, 1)
                writer_id = path.split(os.sep)[-3]
                if writer_id not in writer_pool:
                    writer_pool[writer_id] = []
                writer_pool[writer_id].append(path)
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line or path: {line.strip()}")
                continue
        
        self.data = []
        for writer_id, paths in writer_pool.items():
            if len(paths) > 1:
                for path in paths:
                    self.data.append({"path": path, "writer_id": writer_id, "pool": paths})
        print(f"Found {len(self.data)} images for style encoder training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        anchor_path_relative = item["path"]
        anchor_path_full = os.path.join(self.data_root, anchor_path_relative)
        writer_id = item["writer_id"]
        
        positive_pool = [os.path.join(self.data_root, p) for p in item["pool"] if p != anchor_path_relative]
        positive_path = random.choice(positive_pool)
        
        negative_item = random.choice(self.data)
        while negative_item["writer_id"] == writer_id:
            negative_item = random.choice(self.data)
        negative_path = os.path.join(self.data_root, negative_item["path"])

        anchor_img = Image.open(anchor_path_full).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        fheight, fwidth = self.fixed_size
        def process_image(img):
            if self.subset == 'train':
                nwidth = int(np.random.uniform(0.8, 1.2) * img.width)
                nheight = int((np.random.uniform(0.8, 1.2) * img.height / img.width) * nwidth)
            else:
                nheight, nwidth = img.height, img.width
            
            nheight = max(8, min(fheight - 8, nheight))
            nwidth = max(8, min(fwidth - 8, nwidth))
            resized = image_resize_PIL(img, height=nheight, width=nwidth)
            return centered_PIL(resized, (fheight, fwidth))
        
        anchor_img, positive_img, negative_img = map(process_image, [anchor_img, positive_img, negative_img])

        if self.transforms:
            anchor_img, positive_img, negative_img = map(self.transforms, [anchor_img, positive_img, negative_img])

        return anchor_img, positive_img, negative_img

    @staticmethod
    def collate_fn(batch):
        anchors, positives, negatives = zip(*batch)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# ===================================================================
# ## Model Architecture
# ===================================================================
class ImageEncoder(nn.Module):
    def __init__(self, model_name='mobilenetv2_100', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)

# ===================================================================
# ## Training and Validation Loops
# ===================================================================
def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    loss_meter = AvgMeter()
    pbar = tqdm(loader, desc="Training")
    for anchor, positive, negative in pbar:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        loss = criterion(anchor_out, positive_out, negative_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), anchor.size(0))
        pbar.set_postfix(loss=loss_meter.avg)
    return loss_meter.avg

def val_epoch(loader, model, criterion, device):
    model.eval()
    loss_meter = AvgMeter()
    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss_meter.update(loss.item(), anchor.size(0))
            pbar.set_postfix(loss=loss_meter.avg)
    return loss_meter.avg

# ===================================================================
# ## Main Execution Block
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Style Encoder with Triplet Loss')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='CNN model from timm library')
    parser.add_argument('--train_annotation', type=str, required=True, help='Path to your train.txt annotation file.')
    parser.add_argument('--val_annotation', type=str, required=True, help='Path to your val.txt annotation file.')
    parser.add_argument('--data_root', type=str, required=True, help='The base path to your dataset image folders (e.g., /scratch/.../wordstylist/).')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='./style_models', help='Path to save trained models')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = StyleEncoderDataset(args.train_annotation, args.data_root, fixed_size=(64, 256), transforms=transform, subset='train')
    val_data = StyleEncoderDataset(args.val_annotation, args.data_root, fixed_size=(64, 256), transforms=transform, subset='val')
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("Error: One or both datasets are empty. Please check your annotation files and paths.")
        return

    print(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=StyleEncoderDataset.collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=StyleEncoderDataset.collate_fn)

    model = ImageEncoder(model_name=args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.TripletMarginLoss(margin=0.2)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
        val_loss = val_epoch(loader=val_loader, model=model, criterion=criterion, device=device)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_file = os.path.join(args.save_path, f'wordstylist_{args.model}_best.pth')
            torch.save(model.state_dict(), save_file)
            print(f"✅ Saved Best Model! (Validation Loss: {best_loss:.4f})")
            
    print("Finished training style encoder.")

if __name__ == '__main__':
    main()