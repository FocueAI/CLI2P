import os, torch, json, sys, pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))
sys.path.append(str(__dir__.parent.parent))
# print(f'sys-path:{sys.path}')

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from data import load_dataset
from utils import get_lr
from tqdm import tqdm
from models.cli2p import CLI2P
from data import load_dataset
from dataloader import SiameseDataset, dataset_collate
from models import load_from_name, tokenize
from tools import contrastive
from tools.utils import fit_one_epoch
from tools.callbacks import LossHistory

# step1: 数据加载模块
dataset_path = r'dataset'
train_img_lines, train_text_lines, train_labels, val_img_lines, val_text_lines, val_labels = load_dataset(dataset_path)
# print(f"train_img_lines:{train_img_lines}")
input_shape = [224, 224]
batch_size = 2
## 训练数据
train_dataset  = SiameseDataset(input_shape, train_img_lines, train_text_lines, train_labels, True, autoaugment_flag=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset_collate )
## 训练数据
val_dataset  = SiameseDataset(input_shape, train_img_lines, train_text_lines, train_labels, True, autoaugment_flag=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate )


# step2：模型加载模块
cli2p_model = CLI2P({}) # TODO: 可以跑通该代码
cli2p_model = cli2p_model.cuda()

# step3: 损失函数加载模块
loss_fn = contrastive.ContrastiveLoss()

# step4: 优化器的选择
Init_lr_fit = 3e-5
momentum      = 0.9
weight_decay  = 0
optimizer_type = 'adam'
optimizer = {
    'adam'  : optim.Adam(cli2p_model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
    'sgd'   : optim.SGD(cli2p_model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
}[optimizer_type]


# step5: 
save_dir = r'./model_weight_9_18'



loss_history = LossHistory(save_dir, cli2p_model, input_shape=input_shape)



# step4：学习率调整单元
pass

# step5: 开始训练
Epoch = 100
for epoch in range(Epoch):
    fit_one_epoch(
        model=cli2p_model, 
        train_data_loader = train_dataloader,
        val_data_loader = val_dataloader,
        loss_fn = loss_fn,
        loss_history = loss_history,
        optimizer = optimizer,
        epoch_no=epoch,
        epoch_num =Epoch,
        per_epoch_train_steps= len(train_img_lines) // batch_size,
        per_epoch_val_steps = len(train_img_lines) // batch_size,
        save_weight_dir = save_dir,
        use_cuda = True,
    )








# Epoch = 100
# for epoch in range(Epoch):
#     for imgs, texts, labels in dataloader:
#         print('-'*10)
#         # print('imgs.shape:',imgs.shape)     # torch.Size([2, 4, 3, 512, 1536])
#         # print('texts.shape:',texts.shape)   # texts.shape: (2, 4, 51)
#         # print('label.shape:',labels.shape)  # torch.Size([4, 1])
        
        
#         img_1, img_2 = imgs     # img_1.shape=img_2.shape=(4, 3, 512, 1536)
#         text_1, text_2 = texts  # text_1.shape=text_2.shape=(4,51)
#         # text_1 = cli2p_model.text_preprocessor(text_1)
#         # text_2 = cli2p_model.text_preprocessor(text_2)
#         # print(f"text_1:{text_1}")
#         # print(f"text_2:{text_2}")
#         labels = labels.cuda()
#         img_text_feature1 = cli2p_model(img_1.cuda(), text_1.cuda())
#         img_text_feature2 = cli2p_model(img_2.cuda(), text_2.cuda())


#         print(f'img_text_feature1:{img_text_feature1.shape}')  # .shape=(4, 1024)
#         print(f'img_text_feature2:{img_text_feature2.shape}')  # .shape=(4, 1024)

#         loss = loss_fn((img_text_feature1, img_text_feature2), labels)
        
#         print(f'loss:{loss}')