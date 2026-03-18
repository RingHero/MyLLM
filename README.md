### 小语言模型
跟着网上的教程，结合AI的教导，写了个小型的语言模型。
包含：预训练、SFT、KVCache推理

### 具体文件内容
#### model文件夹里包含模型定义与数据集定义。
##### 模型定义：
model_lm.py是基础的模型定义，无KVcache，主要用于训练
model_lm_forward.py是带KVcache的推理模型，用于推理
##### 数据集：
mydataset.py定义了两种文件的数据集，一个是jsonl，一个是bin(bin也是从jsonl转化的)
prepare_data.py用于将jsonl文件转换为bin文件，训练时更容易加载
以上两个文件专用于预训练部分
SFT_dataset.py定义了两种文件的SFT数据集，一种是jsonl，一种是csv
SFT_chat.py用于一切训练完后进行助理对话（真正意义上的说人话了）

#### 训练流程
定义了三个文件：
trainer.py 单卡jsonl训练
trainerDDP.py 多卡训练，启动命令为
```python
torchrun --nproc-per-node=4 trainerDDP.py
```
trainerSFT_DDP.py 多卡SFT训练
```python
torchrun --nproc-per-node=4 trainerSFT_DDP.py
```
