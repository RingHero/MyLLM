import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset_json(Dataset): 
    #直接用的Jsonl文件作为数据集，但是训练的时候不能把所有数据集导入，对模型训练会不够充分，而且加载起来非常低效
    def __init__(self, path, max_length):

        import tiktoken
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_length = max_length

        self.encoded_data = []

        #特殊符号
        self.bos_token = self.enc.encode('<|startoftext|>', allowed_special={'<|startoftext|>'})[0]
        self.eos_token = self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        #gpt2的分词器没有pad_token
        #self.pad_token = self.enc.encode('<|pad|>', allowed_special={'<|pad|>'})[0]

        #读取文件
        self.max_lines = 10000
        import json
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except Exception as e:
                    pass

        #编码
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token]) 

        #长文本分割
        for i in range(0, len(full_encoded), self.max_length):
            chunk = full_encoded[i:i+self.max_length]
            #填充pad
            if len(chunk) < self.max_length:
                chunk += [self.eos_token] * (self.max_length - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class MyDataset_bin(Dataset):
    # 问了AI给我的一个更好的方案，生成bin文件，随后导入bin文件通过memmap内存映射，不仅能导入全部数据节省内存还能提高速度
    def __init__(self, bin_path, max_length):
        self.max_length = max_length
        # 使用 memmap 映射文件，模式为只读 'r'
        # 注意 dtype 必须和预处理时保持一致 (np.uint16)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        # 计算可以切出多少个完整的片段
        # 因为 x 和 y 需要错开一位，所以每个样本实际需要 max_length + 1 个 token
        self.total_samples = len(self.data) // (self.max_length + 1)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 根据索引算出在二进制数组中的起始位置
        start_idx = idx * (self.max_length + 1)
        end_idx = start_idx + (self.max_length + 1)
        
        # 从硬盘(memmap)中瞬间切出这一小块，转成 int64 的 tensor 给 PyTorch
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        
        # x 取前 max_length 个，y 取后 max_length 个
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y





