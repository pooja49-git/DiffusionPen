# import torch.nn as nn
# import timm

# class ImageEncoder(nn.Module):
#     """
#     Encode images to a fixed size vector
#     """

#     def __init__(
#         self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
#     ):
#         super().__init__()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=num_classes, global_pool="max"
#         )
#         #self.model = torch.compile(self.model, backend="inductor")
#         for p in self.model.parameters():
#             p.requires_grad = trainable
#     def forward(self, x):
#         x = self.model(x)
#         return x   

# import torch
# import torch.nn as nn
# import timm

# class ImageEncoder(nn.Module):
#     """
#     Encode images and style reference images to a fixed size vector
#     """
#     def __init__(self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True):
#         super().__init__()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=num_classes, global_pool="max"
#         )
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#     def forward(self, img, s_imgs):
#         # Encode main image
#         img_feat = self.model(img)  # [B, F]

#         # Encode style images
#         # s_imgs shape: [B, N, C, H, W]
#         B, N, C, H, W = s_imgs.shape
#         s_imgs = s_imgs.view(B * N, C, H, W)  # flatten for batch processing
#         s_feats = self.model(s_imgs)          # [B*N, F]
#         s_feats = s_feats.view(B, N, -1).mean(dim=1)  # average style features

#         # Combine features
#         combined = torch.cat([img_feat, s_feats], dim=1)  # [B, 2F]
#         return combined

import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, model_name='mobilenetv2_100'):
        super().__init__()
        # This creates the pre-trained model but removes the final classification layer.
        self.model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,       # 0 classes = remove classifier
            global_pool="avg"    # Add a global average pooling layer at the end
        )
    
    def forward(self, x):
        # The forward pass just sends the input through the model to get the feature vector.
        return self.model(x)
