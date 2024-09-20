import os, torch, json, sys, pathlib, random, copy
from tqdm import tqdm

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))
sys.path.append(str(__dir__.parent.parent))
from tools.contrastive import ContrastiveLoss
from PIL import Image
from models.cli2p import CLI2P

class Computer_im_text_feature_D:
    """
    计算 图文联合特征之间的距离的类
    """
    def __init__(self, weights_dir=r'./model_weight_9_19') -> None:
        """
        加载权重， 初始化模型示例
        """
        model_path = os.path.join(weights_dir, "best_epoch_weights.pth")
        model_dict = torch.load(model_path)
        self.cli2p_model = CLI2P({}) 
        self.cli2p_model.load_state_dict(model_dict)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cli2p_model = self.cli2p_model.to(self.device)

    def __call__(self, im_text_pair1_path, im_text_pair2_path, label=torch.tensor([1]), return_f = False):
        # 读取第1组 图文对
        pil_img_1 = Image.open(im_text_pair1_path[0])
        with open(im_text_pair1_path[1], 'r', encoding='utf-8') as reader:
            text_1_con = reader.readline().strip()
        # 读取第2组 图文对
        pil_img_2 = Image.open(im_text_pair2_path[0])
        with open(im_text_pair2_path[1], 'r', encoding='utf-8') as reader:
            text_2_con = reader.readline().strip()

        img_1 = self.cli2p_model.img_preprocessor(pil_img_1).unsqueeze(0).to(self.device)
        img_2 = self.cli2p_model.img_preprocessor(pil_img_2).unsqueeze(0).to(self.device)


        text1 = self.cli2p_model.text_preprocessor(text_1_con,context_length=120).to(self.device)
        text2 = self.cli2p_model.text_preprocessor(text_2_con,context_length=120).to(self.device)

        with torch.no_grad():
            mix_feat1 = self.cli2p_model(img_1, text1)
            mix_feat2 = self.cli2p_model(img_2, text2)
            distances = torch.sum((mix_feat1-mix_feat2)**2, dim=1)
            # print(f"distances:{distances}")
            return distances.item(), " "

if __name__ == "__main__":
    im_distance_computer = Computer_im_text_feature_D()
    # ----------------------------------------------------------------------- #    
    img_dir = r"datasets_book_spine/train"  
    all_img_dir = [os.path.join(img_dir, i) for i in os.listdir(img_dir) ]
    random.shuffle(all_img_dir)
    # print(all_img_dir)
    acc_collecter = []
    for i in tqdm(range(len(all_img_dir))):
        all_img_dir_copy = copy.deepcopy(all_img_dir)
        
        # 测试本类 的数据
        this_class_dir =  all_img_dir[i]
        this_class_allfile_path_l = [os.path.join(this_class_dir, i) for i in os.listdir(this_class_dir) if i.endswith(('.jpg','.png'))]
        this_class_allfile_path_l = sorted(this_class_allfile_path_l)
        select_master_img_path = this_class_allfile_path_l[0]
        select_master_text_path = os.path.splitext(select_master_img_path)[0] + '.txt'
        
        select_class_compara_img_path = this_class_allfile_path_l[1]
        select_class_compara_text_path = os.path.splitext(select_class_compara_img_path)[0] + '.txt'
    
        
        may_min_distance, loss = im_distance_computer((select_master_img_path, select_master_text_path), (select_class_compara_img_path, select_class_compara_text_path),return_f=True)
        
        
        print("="*6)
        print(f"select_master_img_path:{select_master_img_path}, select_class_compara_img_path:{select_class_compara_img_path}")
        print(f'may_min_distance:{may_min_distance},loss:{loss}')
        # 
        all_img_dir_copy.pop(i)
        other_cls_master_l = []
        
        for j in all_img_dir_copy:
            dir_ = j  # 其他一个类 的文件夹 路径
            for jj in os.listdir(dir_) :
                if jj.endswith(('.jpg','.png')):
                    img_path = os.path.join(dir_,jj)
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    # if jj.endswith("-0.jpg"):
                    if True:
                        # other_cls_master = Image.open(img_path)
                        other_cls_distance, _ = im_distance_computer((select_master_img_path, select_master_text_path), (img_path, txt_path))
                        # print(f"other_cls_distance:{other_cls_distance}")
                        if may_min_distance > other_cls_distance:
                            # print(f'err.....')
                            other_cls_master_l.append({img_path:other_cls_distance})
                        
                        break # 因为与其他类的 master 图像做距离 计算
        # print(f"other_cls_master_l:{other_cls_master_l}",len(other_cls_master_l))
        this_acc = 1-(len(other_cls_master_l)/len(all_img_dir))
        print(f"top-acc",len(other_cls_master_l), this_acc)
        acc_collecter.append(this_acc)
        print(f'acc-means:{sum(acc_collecter)/len(acc_collecter)}')  


############### 测试结果 ###############
"""
test-acc: 0.94575963
val-acc:  0.9409542871900828
train-acc: 0.940823823225139
"""
#########################################













