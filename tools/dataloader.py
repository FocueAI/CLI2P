import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils import cvtColor, preprocess_input
from utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize, Text_aug
from models import tokenize

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDataset(Dataset):
    def __init__(self, input_shape, img_lines, text_lines, labels, random=False, autoaugment_flag=True, context_length=120):
        self.input_shape     = input_shape
        self.train_img_lines = img_lines     # 所有图像 完整路径             |  [img1_path, img2_path, ... , imgn_path]
        self.train_text_lines= text_lines    # 所有文本 完整路径             |  [txt1_path, txt2_path, ... , txtn_path]
        self.train_labels    = labels        # 所有图像 or 文本 对应的 类别  |   [   0,         1,     ... ,     n    ]
        self.types           = max(labels)   # 总共含有的类别数
        self.context_length  = context_length
        self.random         = random
        
        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
            
        self.text_aug = Text_aug('tools/expore/db/hanzi_similar_list.txt')

    def __len__(self):
        return len(self.train_img_lines)

    def __getitem__(self, index):
        """
        获取 2对 图文对资源： 第1对是相似的图文对资源; 第2对是不相似的图文对资源
        """
        batch_images_path = []
        batch_textes_path = []
        #------------------------------------------#
        #   首先选取三张类别相同的图片
        #------------------------------------------#
        c               = random.randint(0, self.types - 1)                        # 随机选择一个类别
        selected_img_path   = self.train_img_lines[self.train_labels[:] == c]      # 找到该类别的所有图像 路径数据
        selected_text_path   = self.train_text_lines[self.train_labels[:] == c]    # 找到该类别的所有图像 路径数据
        while len(selected_img_path)<3:   # 如果该类别的数据小于3张---------> 就重新选择一个类别  -------------------> 看来每个类别的数据不能太少!!!!!
            c               = random.randint(0, self.types - 1)
            selected_img_path   = self.train_img_lines[self.train_labels[:] == c]
            selected_text_path   = self.train_text_lines[self.train_labels[:] == c]

        selected_indexes = random.sample(range(0, len(selected_img_path)), 3)  # 在该类别中随机 拿出3张图像
        #------------------------------------------#
        #   取出两张类似的图片
        #   对于这两张图片，网络应当输出1  --------------- 获取第一对 图文 对
        #------------------------------------------#
        batch_images_path.append(selected_img_path[selected_indexes[0]])  # 第 1 张  from 第 c 类
        batch_textes_path.append(selected_text_path[selected_indexes[0]]) 
        
        batch_images_path.append(selected_img_path[selected_indexes[1]])  # 第 2 张  from 第 c 类
        batch_textes_path.append(selected_text_path[selected_indexes[1]]) 
        
        #------------------------------------------#
        #   取出两张不类似的图片
        #------------------------------------------#
        batch_images_path.append(selected_img_path[selected_indexes[2]])  # 第 3 张  from 第 c 类
        batch_textes_path.append(selected_text_path[selected_indexes[2]]) 
        #------------------------------------------#
        #   取出与当前的小类别不同的类
        #------------------------------------------#
        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]  # 获取的当前 非 c 类 假定是 b 类
        selected_img_path       = self.train_img_lines[self.train_labels == current_c] # 将当前 b 类的所有 图像路径列表获取到
        selected_text_path       = self.train_text_lines[self.train_labels == current_c]
        while len(selected_img_path)<1:                         # 当 b 类图像 太少的时候。。。。。。。
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_img_path       = self.train_img_lines[self.train_labels == current_c]  # 直到 找到 数量充足 的 非 c 类 为止
            selected_text_path       = self.train_text_lines[self.train_labels == current_c]

        selected_indexes = random.sample(range(0, len(selected_img_path)), 1)
        batch_images_path.append(selected_img_path[selected_indexes[0]]) # 第 4 张  from 第 非c 类
        batch_textes_path.append(selected_text_path[selected_indexes[0]])
        
        images, texts, labels = self._convert_path_list_to_images_and_labels(batch_images_path, batch_textes_path, max_text_len=self.context_length)
        return images, texts, labels

    def _convert_path_list_to_images_and_labels(self, img_path_list, text_path_list,  max_text_len=52):
        #-------------------------------------------#
        #   img_path_list  中有固定的4张图像     [类别a，类别a，类别a，类别b]
        #   text_path_list 中也有固定的4个文本   [类别a，类别a，类别a，类别b]
        #   当文本的长度不满51个字符时， 就在该文本串后面补充 '卍' 符号
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(img_path_list) / 2)  # 也是固定的数字2， 即有 2对 的意思
        #-------------------------------------------#
        #   定义网络的输入图片和标签
        #-------------------------------------------#                                                                                类别a                 类别a                   类别a                    类别b
        pairs_of_images = [np.zeros((number_of_pairs, 3, self.input_shape[0], self.input_shape[1])) for i in range(2)]  # [ [.shape=[通道数=3,h,w],  .shape=[通道数=3,h,w],      [.shape=[通道数=3,h,w], .shape=[通道数=3,h,w]   ]  ====> 里面都为全 0 数据
        pairs_of_texts  = [np.zeros((number_of_pairs, max_text_len), dtype=np.int32)  for _ in range(2)   ]           # [ [.shape=(max_text_len,), .shape=(max_text_len,)],   [.shape=(max_text_len,), .shape=(max_text_len,)]  ====>                                                   ]
        labels          = np.zeros((number_of_pairs, 1))                                                                # .shape=[组数=2，1]

        #-------------------------------------------#
        #   对图片对进行循环
        #   0,1为同一种类，2,3为不同种类
        #-------------------------------------------#
        for pair in range(number_of_pairs): # [0, 1]
            #-------------------------------------------#
            #   将图片填充到输入1中
            #-------------------------------------------#
            image = Image.open(img_path_list[pair * 2])     #  ------------- path_list[0,2]   类别a - 图像
            with open(text_path_list[pair * 2], 'r', encoding='utf-8') as reader:
                text = reader.readline().strip()            #  ----------------------------   类别a - 文本
                if self.random:
                    # 加入了文本数据增强功能!!! 
                    text = self.text_aug(text)
                
            text=tokenize(text,context_length=self.context_length)[0]    
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[0][pair, :, :, :] = image     # pairs_of_images[0].shape = (2, 3, 512, 1536)
            pairs_of_texts[0][pair,  :len(text) ] = text  # pairs_of_texts[0].shape = (2,51)

            #-------------------------------------------#
            #   将图片填充到输入2中
            #-------------------------------------------#
            image = Image.open(img_path_list[pair * 2 + 1])  #  ------------- path_list[1,3]
            with open(text_path_list[pair * 2 + 1], 'r', encoding='utf-8') as reader:
                text = reader.readline().strip()
                if self.random:
                    # 加入了文本数据增强功能!!! 
                    text = self.text_aug(text)
            text=tokenize(text,context_length=self.context_length)[0] 
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[1][pair, :, :, :] = image
            pairs_of_texts[1][pair,  :len(text) ] = text    
            if (pair + 1) % 2 == 0:   # （0+1）%2 == 1  /   (1+1) % 2 == 0
                labels[pair] = 0     # 不同类别 他们的标签为 1
            else:
                labels[pair] = 1     # 相同类别 他们的标签为 0

        #-------------------------------------------#
        #   随机的排列组合
        #-------------------------------------------#
        random_permutation = np.random.permutation(number_of_pairs)  #  由于 number_of_pairs=2， 则 random_permutation=[1,0] or [0,1]
        labels = labels[random_permutation]
        
        ##  pairs_of_images[0] 中的2 张图像都是相似的!!!
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]  #  类别a -图像 ------- 类别a -图像  =============>  2 图像可能互换
        pairs_of_texts[0][:, :]  = pairs_of_texts[0][random_permutation, :] 
        
        
        ##  pairs_of_images[1] 中的2张图形都是 不相似的!!! 
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]  #  类别a -图像 ------- 类别b -图像  =============>  2 图像可能互换
        pairs_of_texts[1][:, :]  =  pairs_of_texts[1][random_permutation, :] 
        
        return pairs_of_images, pairs_of_texts, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:  # val
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image

# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images, left_texts     = [], []
    right_images, right_texts    = [], []
    labels          = []                  # 由于 batch_size设置为4， 所以 len(batch)=4,   
    for pair_imgs, pair_texts, pair_labels in batch:         
        
                                                 #                 left_images(相似)      right_images(不相似)
        for i in range(len(pair_imgs[0])):       # pair_imgs = [.shape=(2,3,512,1536), .shape=(2,3,512,1536)],    pair_labels = [ [1.], [0.] ]  ========= > 可见 当batch=1的时候，就有4张图像
            left_images.append(pair_imgs[0][i])  # 
            left_texts.append(pair_texts[0][i])
            
            right_images.append(pair_imgs[1][i])
            right_texts.append(pair_texts[1][i])
            
            labels.append(pair_labels[i])
    #  len(left_images) = len(right_images) = 8     left_images 都是相似的图像对，   right_images 都是不相似的图像对
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.HalfTensor)  # FloatTensor
    # TODO: 这里的texts 其实也要处理成tensor的形式，但是这里还未引入到 token技术， 暂时先使用 numpy的格式
    texts = torch.from_numpy(np.array([left_texts, right_texts])).type(torch.LongTensor)
    
    labels = torch.from_numpy(np.array(labels)).type(torch.HalfTensor) # FloatTensor
    return images, texts, labels