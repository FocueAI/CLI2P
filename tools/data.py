import math
import os,sys
#############################################################
# print('='*8)
# __dir__ = pathlib.Path(os.path.abspath(__file__))
# print(f'here-path:{__dir__}')
# sys.path.append(str(__dir__.parent.parent))
# sys.path.append(str(__dir__.parent.parent))
#############################################################
import random
from functools import partial
from random import shuffle

import numpy as np
from PIL import Image

from utils_aug import center_crop, resize


def load_dataset(dataset_path, train_ratio=0.9):
    types       = 0
    train_path  =  dataset_path            # os.path.join(dataset_path, 'images_background')
    # train_path  = os.path.join(dataset_path, 'images_background')
    image_path_lines = [] 
    text_path_lines =  []
    labels      =      []
    
    #-------------------------------------------------------------#
    #   自己的数据集，遍历大循环
    #-------------------------------------------------------------#
    for character in os.listdir(train_path):
        #-------------------------------------------------------------#
        #   对每张图片进行遍历 ---- 其中 character 为 最后一级文件夹  里面存放的图像 属于同一种类。。。。。
        #-------------------------------------------------------------#
        character_path = os.path.join(train_path, character)
        for image in os.listdir(character_path):
            raw_name, raw_extend_name = os.path.splitext(image)
            if image.endswith(('.jpg','.png')):
                image_path_lines.append(os.path.join(character_path, image))   # image_path_lines 存放着 图像的完整路径
                assert os.path.exists(os.path.join(character_path, raw_name+'.txt'))
            elif image.endswith('.txt'):
                text_path_lines.append(os.path.join(character_path, image))    # text_path_lines 存放着 图像的完整路径
                assert os.path.exists(os.path.join(character_path, raw_name+'.jpg')) or os.path.exists(os.path.join(character_path, raw_name+'.png'))
            labels.append(types)                                # labels 存放着 图像对应的标签类型 0,1,2,3,4....
        types += 1  # 可见每一个最底层的文件夹，是一个类别
    assert len(image_path_lines) == len(text_path_lines)
    
    #-------------------------------------------------------------#
    #   将获得的所有图像进行打乱。
    #-------------------------------------------------------------#
    random.seed(1)
    shuffle_index = np.arange(len(image_path_lines), dtype=np.int32)
    shuffle(shuffle_index)  # 将索引序号 打乱
    random.seed(None)
    
    # 将所有的数据类型 ===> numpy 
    image_path_lines  = np.array(image_path_lines,dtype=object)  # 图像路径 数组
    text_path_lines   = np.array(text_path_lines,dtype=object)   # ocr识别文本路径 数组
    labels            = np.array(labels)              # 图像类别 数组 [0,1,2,3,4,...]
    
    image_path_lines   = image_path_lines[shuffle_index]   # 打乱之后的 图像路径 数组  
    text_path_lines    = text_path_lines[shuffle_index]   # 打乱之后的 图像路径 数组  
    labels             = labels[shuffle_index]  # 打乱之后的 图像类别 数组 （与图像路径是一一对应关系。。。。。）
    
    #-------------------------------------------------------------#
    #   将训练集和验证集进行划分
    #-------------------------------------------------------------#
    num_train           = int(len(image_path_lines)*train_ratio)
    
    # -------划分出来的验证集部分
    val_img_lines       = image_path_lines[num_train:]
    val_text_lines      = text_path_lines[num_train:]
    val_labels          = labels[num_train:]
    
    # -------划分出来的训练集部分
    train_img_lines   = image_path_lines[:num_train]
    train_text_lines  = text_path_lines[:num_train]
    train_labels      = labels[:num_train]
    
    return train_img_lines, train_text_lines, train_labels, val_img_lines, val_text_lines, val_labels
    # return train_lines, train_labels, val_lines, val_labels

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

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg16'         : 'https://download.pytorch.org/models/vgg16-397923af.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
