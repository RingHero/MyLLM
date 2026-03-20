### 小语言模型
跟着网上的教程，结合AI的教导，写了个小型的语言模型。
包含：预训练、SFT、KVCache推理

### 具体文件内容
#### model文件夹里包含模型定义与数据集定义。
##### 模型定义：
model_lm.py是基础的模型定义，无KVcache，主要用于训练
model_lm_forward.py是带KVcache的推理模型，用于推理
model_lora.py定义了lora模型（借鉴了minimind项目的写法）
##### 数据集：
mydataset.py定义了两种文件的数据集，一个是jsonl，一个是bin(bin也是从jsonl转化的)，该项目采用了**seq_monkey**数据集作为预训练数据集。

prepare_data.py用于将jsonl文件转换为bin文件，训练时更容易加载
以上两个文件专用于预训练部分

SFT_dataset.py定义了两种文件的SFT数据集，一种是jsonl，一种是csv
SFT_chat.py用于一切训练完后进行助理对话（真正意义上的说人话了）,该项目使用了**MiniMind提供的sft_1024.jsonl**数据集

DPO_dataset.py定义了DPO数据集，用于RLHF 所用的数据为**HF上的hh_rlhf_train.jsonl**

#### 训练流程
trainer.py 单卡jsonl预训练
```python
python trainer\trainer.py
```
trainerDDP.py 多卡训练（基座预训练）
```python
torchrun --nproc-per-node=4 trainer\trainerDDP.py
```
trainerSFT_DDP.py 多卡SFT训练（全量微调）
```python
torchrun --nproc-per-node=4 trainer\trainerSFT_DDP.py
```
trainerLoRADDP.py 多卡LoRA微调(在model_lora里修改lora_rank)
```python
torchrun --nproc-per-node=4 trainer\trainerLoRADDP.py
```
trainerDPO.py 多卡DPO训练
```python
torchrun --nproc-per-node=4 trainer\trainerDPO.py
```

#### 推理流程
SFT_chat.py SFT推理
```python
python SFT_chat.py
```
lora_chat.py LoRA推理
```python
python lora_chat.py
```
DPO后的推理直接用SFT_chat.py推理


#### 参考
[UP主@chaofa用代码打点酱油](https://www.bilibili.com/video/BV1qWwke5E3K/?spm_id_from=333.337.search-card.all.click&vd_source=4a26d77ea89445c8035479cb5957a79f)

[项目@MiniMind](https://github.com/jingyaogong/minimind)

#### TODO:
- PagedAttention
- MQA、GQA、MLA
