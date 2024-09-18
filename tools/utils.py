import math
import os
import random
from functools import partial
from random import shuffle

import numpy as np
from PIL import Image

from utils_aug import center_crop, resize
import torch
from tqdm import tqdm


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
                  is_fp16=False,     # 是否使用 fp16 精度
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
        # images_bank, texts_bank = batch[0], batch[1]
        # images_cost, texts_cost = batch[2], batch[3]
        imgs, texts, labels = batch
        img_1, img_2 = imgs 
        text_1, text_2 = texts
        
        
        if use_cuda:
            img_1 = img_1.cuda(local_rank)
            img_2 = img_2.cuda(local_rank)
            
            text_1 = text_1.cuda(local_rank)
            text_2 = text_2.cuda(local_rank)
            
            labels = labels.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#    
        optimizer.zero_grad()
        if not is_fp16:
            mix_feat_bank = model(img_1, text_1)
            mix_feat_cost = model(img_2, text_2)
            output = loss_fn((mix_feat_bank, mix_feat_cost), labels)
            
            output.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        else:
            from torch.cuda.amp import autocast
            with autocast():
                mix_feat_bank = model(img_1, text_1)
                mix_feat_cost = model(img_2, text_2)
        
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
        
        # images_bank, texts_bank = batch[0], batch[1]
        # images_cost, texts_cost = batch[2], batch[3]
        
        imgs, texts, labels = batch
        img_1, img_2 = imgs 
        text_1, text_2 = texts
        
        
        
        
        
        with torch.no_grad():
            if use_cuda:
                img_1 = img_1.cuda(local_rank)
                img_2 = img_2.cuda(local_rank)
                
                text_1 = text_1.cuda(local_rank)
                text_2 = text_2.cuda(local_rank)
                
                labels = labels.cuda(local_rank)
    
            optimizer.zero_grad()
            mix_feat_bank = model(img_1, text_1)
            mix_feat_cost = model(img_2, text_2)
            output = loss_fn((mix_feat_bank, mix_feat_cost), labels)
        
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






#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    x /= 255.0
    return x

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

