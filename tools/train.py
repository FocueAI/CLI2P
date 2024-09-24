import os, torch, json, sys, pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))
sys.path.append(str(__dir__.parent.parent))
# print(f'sys-path:{sys.path}')

import numpy as np
import torch, yaml
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from data import load_dataset
from utils import get_lr
from tqdm import tqdm
from models.cli2p import CLI2P
from data import load_dataset
from dataloader import SiameseDataset, dataset_collate
from models import load_from_name, tokenize, image_transform
from tools import contrastive
from tools.utils import fit_one_epoch
from tools.callbacks import LossHistory

################################## ------------------------------------- 总体参数设置区域 begin ----------------------------------------- ########################################
with open('train_configs/normal_args.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
config_global = config['Global']
################################## ------------------------------------- 总体参数设置区域 end ----------------------------------------- ########################################


ngpus_per_node  = torch.cuda.device_count()
if config_global['distributed']:
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    rank        = int(os.environ["RANK"])
    device      = torch.device("cuda", local_rank)
    if local_rank == 0:
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
        print("Gpu Device Count : ", ngpus_per_node)
else:
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    rank            = 0    




# step1: 数据加载模块

train_img_lines, train_text_lines, train_labels= load_dataset(config_global["train_dataset_path"])
val_img_lines, val_text_lines, val_labels = load_dataset(config_global["val_dataset_path"])


# print(f"train_img_lines:{train_img_lines}")
##### random = True 图像进行复杂的图像增强， =False 图像仅仅做常规的resize处理，文本不做数据增强
## 训练数据
train_dataset  = SiameseDataset(config_global['input_shape'], train_img_lines, train_text_lines, train_labels, random=True, autoaugment_flag=False,context_length=config_global["context_length"])
train_dataloader = DataLoader(train_dataset, batch_size=config_global['batch_size'], shuffle=True, collate_fn=dataset_collate, num_workers=config_global['num_workers'], pin_memory=True)
## 训练数据
val_dataset  = SiameseDataset(config_global['input_shape'], val_img_lines, val_text_lines, val_labels, random=False, autoaugment_flag=False, context_length=config_global["context_length"])
val_dataloader = DataLoader(val_dataset, batch_size=config_global['batch_size'], shuffle=False, collate_fn=dataset_collate, num_workers=config_global['num_workers'], pin_memory=True)


# step2：模型加载模块
config = {
    "freeze_flag": True,
    "visual_freeze_last_layers": 14,
    "bert_freeze_last_layers": 16
}
cli2p_model = CLI2P(**config) 
if config_global['use_cuda']:
    cli2p_model = cli2p_model.cuda()

if not os.path.exists(config_global['save_dir']):
    os.makedirs(config_global['save_dir'])

model_path = os.path.join(config_global['save_dir'], "best_epoch_weights.pth")
if config_global['pretrained'] and os.path.exists(model_path):
        cli2p_model_dict      = cli2p_model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in cli2p_model_dict.keys() and np.shape(cli2p_model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        cli2p_model_dict.update(temp_dict)
        cli2p_model.load_state_dict(cli2p_model_dict)
        print(f'no_load_key:{no_load_key}')

# step3: 损失函数加载模块
loss_fn = contrastive.ContrastiveLoss()

# step4: 优化器的选择
Init_lr_fit = 3e-5
momentum      = 0.9
weight_decay  = 0
optimizer = {
    'adam'  : optim.Adam(cli2p_model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
    'sgd'   : optim.SGD(cli2p_model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
}[config_global['optimizer_type']]


# step5: 
loss_history = LossHistory(config_global['save_dir'], cli2p_model, input_shape=config_global['input_shape'])



# step6：学习率调整单元


lr_schedular = {
    'StepLR': StepLR(optimizer=optimizer,step_size=3,gamma=0.9), # 每 2 轮 学习率 变成原来的 0.9倍
    'ReduceLROnPlateau': ReduceLROnPlateau(optimizer=optimizer, mode='min',factor=0.9, patience=3, eps=1e-10, threshold=1e-3) # 当学习率小于eps之后，学习率将不在调整!!!
    
}[config_global['lr_schedular_type']]




# step7: 开始训练
Epoch = config_global["num_epoch"]
for epoch in range(Epoch):
    val_loss = fit_one_epoch(
        model=cli2p_model, 
        train_data_loader = train_dataloader,
        val_data_loader = val_dataloader,
        loss_fn = loss_fn,
        loss_history = loss_history,
        optimizer = optimizer,
        epoch_no=epoch,
        epoch_num =Epoch,
        per_epoch_train_steps= len(train_img_lines) // config_global['batch_size'],
        per_epoch_val_steps = len(val_img_lines) // config_global['batch_size'],
        save_weight_dir = config_global['save_dir'],
        use_cuda = config_global['use_cuda'],
    )
    if config_global['lr_schedular_type'] == 'ReduceLROnPlateau':
        lr_schedular.step(val_loss)
    else:
        lr_schedular.step()
loss_history.writer.close()