# 图文联合搜索 💻   
<h3> 🔥 Contrastive Language-img-Language-img-Pre-training , 简称CLI2P</h3>  

## 模型架构图  
<!-- ![模型](./corpus/image.png) -->
<img src="./corpus/image.png" width="500px" height="300">

### TODO-LIST：
<!--  ✅ ❌ -->
- ✅ 模型搭建(可以在test_cli2p.ipynb中体验其前向传播) -----已经经过简单验证通过(还是个毛坯房)
- ✅ 数据加载单元 
- ✅ 模型训练代码 
- ✅ 模型损失代码  
- ✅ 模型推理代码 
- ✅ 模型转换代码  （predict_with_acc.py 完成模型转换 ------ predict_with_acc_jit.py 完成转换后模型的使用）    
- ✅ 引入faiss模块  https://github.com/FocueAI/CLI2P-inference-with-faiss.git
- ❌ 博客介绍

### 数据准备：
在该工程的根目录下建立一个 文件夹    
-datasets/   
------类型1/  
---------图像1  
---------图像1的ocr结果   
---------...   
---------图像n   
---------图像n的ocr结果   
------...   
------类型n/  
---------图像1  
---------图像1的ocr结果  
---------...   
---------图像n   
---------图像n的ocr结果  


参考：https://github.com/OFA-Sys/Chinese-CLIP
