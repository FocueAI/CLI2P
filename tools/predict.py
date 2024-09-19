import os, torch, json, sys, pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))
sys.path.append(str(__dir__.parent.parent))

from PIL import Image

from models.cli2p import CLI2P
cli2p_model = CLI2P({}) 
save_dir = r'./model_weight_9_18'
model_path = os.path.join(save_dir, "best_epoch_weights.pth")
model_dict = torch.load(model_path)
cli2p_model.load_state_dict(model_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
cli2p_model = cli2p_model.to(device)

# 第 1 组 图文对
img_1_path = r'dataset\001T110027850\001T110027850.jpg-0enhance-0.jpg'
text_1_path = r'dataset\001T110027850\001T110027850.jpg-0enhance-0.txt'

# 第 2 组 图文对
img_2_path =  r'dataset\00000941\00000941.jpg-0enhance-0.jpg'
text_2_path = text_1_path # r'dataset\00000941\00000941.jpg-0enhance-0.txt'

pil_img_1 = Image.open(img_1_path)
with open(text_1_path, 'r', encoding='utf-8') as reader:
    text_1_con = reader.readline().strip()
    
pil_img_2 = Image.open(img_2_path)
with open(text_2_path, 'r', encoding='utf-8') as reader:
    text_2_con = reader.readline().strip()


img_1 = cli2p_model.img_preprocessor(pil_img_1).unsqueeze(0).to(device)
img_2 = cli2p_model.img_preprocessor(pil_img_2).unsqueeze(0).to(device)


text1 = cli2p_model.text_preprocessor(text_1_con).to(device)
text2 = cli2p_model.text_preprocessor(text_2_con).to(device)

with torch.no_grad():
    mix_feat1 = cli2p_model(img_1, text1)
    mix_feat2 = cli2p_model(img_2, text2)
    distances = torch.sum((mix_feat1-mix_feat2)**2, dim=1)
    print(f"distances:{distances}")