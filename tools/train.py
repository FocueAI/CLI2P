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

dataset_path = r'dataset'
train_img_lines, train_text_lines, train_labels, val_img_lines, val_text_lines, val_labels = load_dataset(dataset_path)
# print(f"train_img_lines:{train_img_lines}")
input_shape = [224, 224]

# (self, input_shape, img_lines, text_lines, labels, random=False, autoaugment_flag=True):
train_dataset  = SiameseDataset(input_shape, train_img_lines, train_text_lines, train_labels, True, autoaugment_flag=False)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=dataset_collate )
cli2p_model = CLI2P({}) # TODO: 可以跑通该代码
cli2p_model = cli2p_model.cuda()

# def 



for imgs, texts, labels in dataloader:
    print('-'*10)
    print('imgs.shape:',imgs.shape)     # torch.Size([2, 4, 3, 512, 1536])
    print('texts.shape:',texts.shape)   # texts.shape: (2, 4, 51)
    print('label.shape:',labels.shape)  # torch.Size([4, 1])
    
    
    img_1, img_2 = imgs     # img_1.shape=img_2.shape=(4, 3, 512, 1536)
    text_1, text_2 = texts  # text_1.shape=text_2.shape=(4,51)
    # text_1 = cli2p_model.text_preprocessor(text_1)
    # text_2 = cli2p_model.text_preprocessor(text_2)
    print(f"text_1:{text_1}")
    print(f"text_2:{text_2}")
    
    img_text_feature1 = cli2p_model(img_1.cuda(), text_1.cuda())
    img_text_feature2 = cli2p_model(img_2.cuda(), text_2.cuda())


    print(f'img_text_feature1:{img_text_feature1}')
    print(f'img_text_feature2:{img_text_feature2}')

