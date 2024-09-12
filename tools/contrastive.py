import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/804463592/pytorch_siamese_minist/blob/master/contrastive.py

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, x, labels, train_flag=True):
        x0, x1 = x  # x0.shape = [8,512]
        distances = torch.sum((x0-x1)**2, dim=1)
        labels = labels.reshape(-1)
        # losses = (1 - labels) * distances + labels * F.relu(self.margin - distances)
        if train_flag:
            losses = labels * distances + (1 - labels) * F.relu(self.margin - distances) * 5
        else:
            losses = labels * distances + (1 - labels) * F.relu(self.margin - distances)
        # 只考虑有效的损失
        # print(f'loss1:{losses}')
        losses = losses.clamp(min=0)
        
        loss = torch.mean(losses)
        # print(f'loss:{loss}')
        return loss