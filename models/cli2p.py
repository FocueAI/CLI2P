import torch
import torch.nn as nn
from models.model import CLIP
from models import load_from_name, tokenize
from models.feature_mix import MultiModalTransformer
class CLI2P(nn.Module):
    _defaults  = {
        "device": "cpu",
        "model_name": "ViT-B-16",
        "device": 'cuda',
        "download_root": './',   # 如果下载模型权重，应该保存的路径
        "mask_ratio": 0.3
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        # self.feature_mix = nn.Sequential(
        #     # 自编码器-编码器部分
        #     nn.Linear(1024, 2048),  # 将1024维输入编码到512维潜在空间
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),  # 添加 Dropout
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),  # 添加 Dropout
        #     nn.Linear(1024, 512),  # 潜在空间维度为128
        #     # 自编码器-解码器部分
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),  # 添加 Dropout
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),  # 添加 Dropout
        #     nn.Linear(2048, 1024),  # 将潜在空间重构回1024维
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),  # 添加 Dropout
        #     nn.Linear(1024, 1024),
        #     # nn.Sigmoid() 
            
        # ) # .to(torch.float16)
        
        self.__dict__.update(self._defaults) # 新更新默认参数
        self.__dict__.update(kwargs)     # 在更新传入参数
        self.feature_mix = MultiModalTransformer()
        # 图文提取器            图像前处理器
        self.feat_extrator, self.img_preprocessor = load_from_name(self.model_name, device=self.device, download_root=self.download_root, freeze_kwargs=kwargs)
        self.text_preprocessor = tokenize
        # self._initialize_weights()  # 这个是含必要的
    def _initialize_weights(self):
        import torch.nn.init as init
        for m in self.feature_mix.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 也可以使用   
        
    
    def forward(self, image, text):
        """
            image: tensor格式 eg: .shape=[batch=1,3,224,224]
            text:  tensor格式 eg: .shape=[batch=1, seq_max_len=52]
        """

        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.feat_extrator.encode_text(text,)
        elif text is None:
            return self.feat_extrator.encode_image(image, self.mask_ratio)
        # [4,512],       [4, 197, 768]
        image_features, image_features_3d = self.feat_extrator.encode_image(image, self.mask_ratio) # image.shape=torch.Size([4, 3, 224, 224])
        # [4,512],      [4, 120, 768]
        text_features, text_features_3d = self.feat_extrator.encode_text(text) # text.shape=torch.Size([4, 120])
        # ------------------ 使用自编码器的特征融合方法
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True) # .shape=(1, 512)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # .shape=(1, 512)
        # image_text_features = torch.cat((image_features, image_features),dim=1)   # .shape=(1,1024)
        # image_text_features = self.feature_mix(image_text_features.to(torch.float32)) # .shape=(1,1024)   # 会将数据变为nan
        # ------------------ 使用transformer的特征融合方法
        image_text_features = self.feature_mix(text_features_3d,image_features_3d)
        
        
        return image_text_features