import csv
import torch
from torch.utils.data import Dataset
import tiktoken

# 写了两种SFT数据集，一种是从csv文件中读取，一种是从json文件中读取


class SFTDataset_csv(Dataset):
    def __init__(self, csv_path, max_length):
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_length = max_length
        self.eos_token = self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        
        # GPT2 词表没有专门的 pad_token，用 eos_token 占位
        self.pad_token = self.eos_token 
        
        self.data = []
        
        print("正在加载 CSV 格式的 SFT 数据集...")
        # 【修改点 1】使用 csv 模块读取文件
        with open(csv_path, 'r', encoding='utf-8') as f:
            # DictReader 会自动把第一行识别为列名（instruction, input, output）
            reader = csv.DictReader(f)
            
            for row in reader:
                # 获取数据，如果没有就默认为空字符串
                instruction = row.get('instruction', '').strip()
                input_text = row.get('input', '').strip()
                output_text = row.get('output', '').strip()
                
                # 跳过完全为空的异常行
                if not instruction and not output_text:
                    continue
                
                # 【修改点 2】拼接提示词格式
                if input_text:
                    prompt = f"User: {instruction}\n{input_text}\n\nAssistant: "
                else:
                    prompt = f"User: {instruction}\n\nAssistant: "
                    
                answer = f"{output_text}"
                
                self.data.append({"prompt": prompt, "answer": answer})
                
        print(f"SFT 数据集加载完成！共 {len(self.data)} 条高质量问答对。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 分别将 prompt 和 answer 转换为 ID
        prompt_ids = self.enc.encode(item["prompt"])
        
        # 回答的最后加上结束符，让模型学会“闭嘴”
        answer_ids = self.enc.encode(item["answer"]) + [self.eos_token] 
        
        input_ids = prompt_ids + answer_ids
        
        # 构造 Labels，利用 -100 屏蔽 prompt 部分的梯度
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # 截断与填充 (Padding)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token] * pad_len
            labels = labels + [-100] * pad_len  # 填充部分也设为 -100
            
        # 错位 1 个 Token，用于 Next Token Prediction
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        
        return x, y

import json

class SFTDataset_json(Dataset):
    def __init__(self, jsonl_path, max_length):
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_length = max_length
        self.eos_token = self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        
        # GPT2 词表没有专门的 pad_token，用 eos_token 占位
        self.pad_token = self.eos_token 
        
        self.data = []
        
        print("正在解析高级对话格式 (JSONL) SFT 数据集...")
        
        # 逐行读取 JSONL 文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line)
                    conversations = item.get("conversations", [])
                    
                    # 确保这行数据至少包含一问一答，并且角色对得上
                    if len(conversations) >= 2 and conversations[0]["role"] == "user" and conversations[1]["role"] == "assistant":
                        user_content = conversations[0]["content"].strip()
                        assistant_content = conversations[1]["content"].strip()
                        
                        # ⚠️ 核心魔法：把 JSON 的角色映射回我们熟悉的“暗号”
                        prompt = f"User: {user_content}\n\nAssistant: "
                        answer = f"{assistant_content}"
                        
                        self.data.append({"prompt": prompt, "answer": answer})
                        
                except Exception as e:
                    # 如果某一行 JSON 损坏，跳过它，防止整个训练崩溃
                    continue
                
        print(f"SFT 数据集加载完成！共成功提取 {len(self.data)} 条高质量问答对。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
       # 分别将 prompt 和 answer 转换为 ID，允许包含所有特殊 Token
        prompt_ids = self.enc.encode(item["prompt"], allowed_special="all")
        
        # 回答的最后加上结束符，让模型学会“闭嘴”
        answer_ids = self.enc.encode(item["answer"], allowed_special="all") + [self.eos_token] 
        
        input_ids = prompt_ids + answer_ids
        
        # 构造 Labels，利用 -100 屏蔽 prompt 部分的梯度
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # 截断与填充 (Padding)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token] * pad_len
            labels = labels + [-100] * pad_len
            
        # 错位 1 个 Token，用于 Next Token Prediction
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        
        return x, y
