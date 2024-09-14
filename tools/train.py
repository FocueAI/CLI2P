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
from dataloader import SiameseDataset


dataset_path = r'dataset'
train_img_lines, train_text_lines, train_labels, val_img_lines, val_text_lines, val_labels = load_dataset(dataset_path)
# print(f"train_img_lines:{train_img_lines}")
input_shape = [512, 1536]

# (self, input_shape, img_lines, text_lines, labels, random=False, autoaugment_flag=True):
train_dataset  = SiameseDataset(input_shape, train_img_lines, train_text_lines, train_labels, True, autoaugment_flag=False)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
for imgs, texts, labels in dataloader:
    print(imgs.shape)



# cli2p_model = CLI2P({}) # TODO: 可以跑通该代码







def fit_one_epoch(model,              # 模型\
                  train_data_loader, # 训练数据加载器
                  val_data_loader,   # 验证数据加载器
                  loss_fn,     # 损失函数
                  loss_history,
                  optimizer,   # 优化器
                  epoch_no,    # 当前训练的世代序号
                  epoch_num,   # 该项目总共训练的世代数
                  per_epoch_train_steps, # 每一个世代，对应的训练步数
                  per_epoch_val_steps,   # 每一个世代，对应的评估步数
                  save_weight_dir, # 模型要保存的文件夹
                  use_cuda,    # 是否使用GPU
                  is_fp16,     # 是否使用 fp16 精度
                  local_rank=0 # 对应的显卡号，如果是DDP模式下，也就是对应的线程号
                  ):
    
    # 定义评价指标等
    train_loss, train_accuracy = 0, 0 
    val_loss, val_accuracy = 0, 0
    
    if local_rank == 0:
        print('Start Train')
        #                反应当轮的进度                         反应了轮次                   用               更新间隔 0.3s
        pbar = tqdm(total=per_epoch_train_steps, desc=f'Epoch {epoch_no + 1}/{epoch_num}', postfix=dict, mininterval=0.3)
        
    # 进入训练模式
    model.train()
    for iteration, batch in enumerate(train_data_loader):
        if iteration >= per_epoch_train_steps:
            break
        images_bank, texts_bank = batch[0], batch[1]
        images_cost, texts_cost = batch[2], batch[3]
        if use_cuda:
            images = images.cuda(local_rank)
            texts = texts.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#    
        optimizer.zero_grad()
        if not is_fp16:
            mix_feat_bank = model(images_bank, texts_bank)
            mix_feat_cost = model(images_cost, texts_cost)
            output = loss_fn(mix_feat_bank, mix_feat_cost)
            
            output.backward()
            output.step()
            
        else:
            from torch.cuda.amp import autocast
            with autocast():
                mix_feat_bank = model(images_bank, texts_bank)
                mix_feat_cost = model(images_cost, texts_cost)
        
        train_loss += output.item() 
        
        # 当 step 结束后， 总结开始了
        if local_rank == 0:
            pbar.set_postfix(**{ 'total_loss': train_loss / (iteration+1),
                'lr': get_lr(optimizer)
                
            })
            pbar.update(1)
            
    # 当一个世代结束时     
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Val')
        #                反应当轮的进度                         反应了轮次                   用               更新间隔 0.3s
        pbar = tqdm(total=per_epoch_val_steps, desc=f'Epoch {epoch_no + 1}/{epoch_num}', postfix=dict, mininterval=0.3)
    # 进入测试模型
    model.eval()
    for iteration, batch in enumerate(val_data_loader):
        if iteration >= per_epoch_val_steps:
            break
        
        images_bank, texts_bank = batch[0], batch[1]
        images_cost, texts_cost = batch[2], batch[3]
        with torch.no_grad():
            if use_cuda:
                images = images.cuda(local_rank)
                texts = texts.cuda(local_rank)   
    
            optimizer.zero_grad()
            mix_feat_bank = model(images_bank, texts_bank)
            mix_feat_cost = model(images_cost, texts_cost)
            output = loss_fn(mix_feat_bank, mix_feat_cost)
        
        # this step finish
        val_loss += output.item()
        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
            })  
            pbar.update(1)
    # this val epoch finish
    if local_rank == 0:
        pbar.close()
        print('Finish val')
        
        loss_history.append_loss(epoch_no +1, train_loss / per_epoch_train_steps, val_loss / per_epoch_val_steps)
        print(f"total-epoch:{epoch_num}, this epoch:{epoch_no+1}")
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss / per_epoch_train_steps, val_loss / per_epoch_val_steps))
        
        # --------------------------------- 保存权重 --------------------------------
        if len(loss_history.val_loss) <= 1 or (val_loss / per_epoch_val_steps) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_weight_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_weight_dir, "last_epoch_weights.pth"))
        
        