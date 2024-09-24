# Code modified from https://github.com/openai/CLIP

import json
import os
from pathlib import Path
from typing import Union, List
import urllib

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from tqdm import tqdm

from models import _tokenizer
from models.model import convert_weights, CLIP, restore_model

__all__ = ["load", "tokenize", "available_models", "image_transform", "load_from_name"]

_MODELS = {
    "ViT-B-16": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt",
    "ViT-L-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14.pt",
    "ViT-L-14-336": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14-336.pt",
    "ViT-H-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt",
    "RN50": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt",
}
_MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14-336": {
        "struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 336
    },
    "ViT-H-14": {
        "struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_from_name(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                   download_root: str = None, vision_model_name: str = None, text_model_name: str = None, input_resolution: int = None, freeze_flag: bool = True):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_name, model_input_resolution = _MODEL_INFO[name]['struct'], _MODEL_INFO[name]['input_resolution']
    elif os.path.isfile(name):
        assert vision_model_name and text_model_name and input_resolution, "Please specify specific 'vision_model_name', 'text_model_name', and 'input_resolution'"
        model_path = name
        model_name, model_input_resolution = f'{vision_model_name}@{text_model_name}', input_resolution
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        # loading saved checkpoint
        checkpoint = torch.load(opened_file, map_location="cpu")

    model = create_model(model_name, checkpoint)  # 将float32 -----> float16 的参数转换给关掉了!!!!!!
    if str(device) == "cpu":
        model.float()
    else:
        model.to(device)
    if freeze_flag:
        # ---------------------------------------------------------------- 冻结参数 -- 视觉
        visual_total_layers = len(list(model.visual.parameters()))
        # 计算最后冻结层的起始索引
        visual_freeze_layers_start = max(0, visual_total_layers - 14)
        
        for i, param in enumerate(model.visual.parameters()):
            if i < visual_freeze_layers_start:     # 之前的层 给冻结住
                param.requires_grad = False  
            else:
                param.requires_grad = True 
        # # 确保最后三层可以训练
        # for param in model.visual.parameters()[freeze_layers_start:]:
        #     param.requires_grad = True
    
        
        # ----------------------------------------------------------------- 冻结参数 -- bert
        bert_total_layers = len(list(model.bert.parameters()))
        # 计算最后冻结层的起始位置
        bert_freeze_layers_start = max(0, bert_total_layers - 16)
        for i, param in enumerate(model.bert.parameters()):
            if i < bert_freeze_layers_start:     # 之前的层 给冻结住
                param.requires_grad = False  
            else:
                param.requires_grad = True 
        
        
        
        
        
        
         
         
        # for param in model.bert.parameters():
        #     pass         
        
        
        
        # for param in model.parameters():
        #     param.requires_grad = False
    return model, image_transform(model_input_resolution)


def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None,
         bert_path=None, use_flash_attention=False):
    """Load CLIP and BERT model weights
    """

    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict, use_flash_attention).to(device)

    if str(device) == "cpu":
        model.float()
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []  # 当 texts=["杰尼龟"] ===============> all_tokens=[[101, 3345, 2225, 7991, 102]]
    for text in texts: # _tokenizer.vocab['[CLS]']=101， _tokenizer.vocab['[SEP]']=102，   
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)   # .shape=[1, 52]

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform


def create_model(model_name, checkpoint=None):
    vision_model, text_model = model_name.split('@')   # vision_model = 'ViT-B-16',  text_model = 'RoBERTa-wwm-ext-base-chinese'
    # ------------------------------------------- 图像编码器的相关配置 ---------------------------------------- # 
    vision_model_config_file = Path(
        __file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)  # WindowsPath('e:/explore/CLI2P/models/model_configs/ViT-B-16.json')
    assert os.path.exists(vision_model_config_file)
    # ------------------------------------------- 文本编码器的相关配置 ---------------------------------------- # 
    text_model_config_file = Path(
        __file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)   # WindowsPath('e:/explore/CLI2P/models/model_configs/RoBERTa-wwm-ext-base-chinese.json')
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])
    print('Model info', model_info)
    model = CLIP(**model_info) # 引入关键性 模型
    # convert_weights(model)   # 将float32 -----> float16
    if checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
    return model
