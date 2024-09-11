import torch
import torch.nn as nn
from model import CLIP

class CLI2P(CLIP):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.feature_mix = nn.Sequential(
            # 自编码器-编码器部分
            nn.Linear(1024, 512),  # 将1024维输入编码到512维潜在空间
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # 潜在空间维度为128
            # 自编码器-解码器部分
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),  # 将潜在空间重构回1024维
            nn.Sigmoid() 
            
        )
    
    def forward(self, image, text):
        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image, mask_ratio)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # .shape=(1, 512)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # .shape=(1, 512)
        
        
        image_text_features = torch.cat((image_features, image_features),dim=1)   # .shape=(1,1024)
        
        image_text_features = self.feature_mix(image_text_features) # .shape=(1,1024)
        
        return image_text_features