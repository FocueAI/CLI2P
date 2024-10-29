# 打算仿写 Q-former 实现该逻辑、
import torch
from torch import nn
# from transformers import BertModel

class MultiModalTransformer(nn.Module):
    def __init__(self,  hidden_dim=768, num_classes=1024):
        super(MultiModalTransformer, self).__init__()
        # self.modality_dim = modality_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Transformer层，输入维度为(sequence_length, batch_size, hidden_dim)
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            # dropout=0.1,
            dropout=0.5,
        )
        self.encoder = nn.TransformerEncoder(self.transformer_layers , num_layers=8)
        # 位置编码，维度为(max_position_embeddings, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        # 输出层，将Transformer的输出映射到分类数量
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, text_features, image_features):
        """
        text_features: [bs=4, len=120, d_model=768] 
        im_features:   [bs=4, len=197, d_model=768]	
        """
        # image_features = image_features[:,0,:] # [bs=4, d_model=768]	
        # 合并文本和图像特征，假设图像特征被复制以匹配文本序列的长度
        # 输出维度为(batch_size, sequence_length + 1, hidden_dim)  经过下一行的代码， cat中的2个元素的维度，分别都为： .shape =(4, 120, 768)
        # combined_features = torch.cat((text_features, image_features.unsqueeze(1).repeat(1, text_features.size(1), 1)), dim=1)  # 在长度上做拼接 .shape=(2, 240, 768)
        combined_features = torch.cat((text_features, image_features), dim=1) # [bs=4, len=197+120=317, d_model=768]	
        
        
        
        # 添加位置编码，位置编码的维度为(1, sequence_length + 1, hidden_dim)
        position_ids = torch.arange(combined_features.size(1), device=combined_features.device)  # .shape=(240,)
        position_embeddings = self.position_embedding(position_ids) # .shape=(240, 786)
        combined_features += position_embeddings                    # .shape=(4, 240, 768)
        
        # 通过Transformer层，输入维度为(sequence_length + 1, batch_size, hidden_dim)
        # transformer_output = self.transformer_layers(combined_features.transpose(0, 1)).transpose(0, 1) # .shape=(4, 240, 768)
        transformer_output = self.encoder(combined_features.transpose(0, 1)).transpose(0, 1)  
        
        
        # 特征融合，取Transformer输出的第一个token（分类token）作为序列的表示
        # 输出维度为(batch_size, hidden_dim)
        pooled_output = transformer_output[:, 0, :]  # .shape=(4,768)
        
        # 通过输出层得到最终分类结果，输出维度为(batch_size=4, num_classes=1024)
        logits = self.output_layer(pooled_output)
        
        return logits
if __name__ == "__main__":
    # 模型初始化
    hidden_dim = 768   # 隐藏层维度，与BERT的维度相同
    num_classes = 1024    # 分类类别数
    model = MultiModalTransformer(hidden_dim, num_classes)

    # 假设文本输入和图像特征
    text_inputs = torch.rand(4, 120,768)  # 假设的文本输入，维度为(batch_size, sequence_length)   .shape=(2,128)
    image_features = torch.rand(4, 197, 768)  # 假设的图像特征，维度为(batch_size, modality_dim)       .shape=(2,512)

    # 前向传播
    outputs = model(text_inputs, image_features)
    print(outputs.shape)  # 输出维度为(batch_size, num_classes)
