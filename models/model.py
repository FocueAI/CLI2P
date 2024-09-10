import torch
import torch.nn as nn





########### 模型的主体部分 ############
class CLI2P(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CLI2P, self).__init__(*args, **kwargs)
        
    