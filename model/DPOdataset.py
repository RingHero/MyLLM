import json
import torch
from torch.utils.data import Dataset
import tiktoken

class DPODataset(Dataset):
    def __init__(self, jsonl_path, max_length):
        self.enc = tiktoken.get_encoding('gpt2')
        self.max_length = max_length
        # 允许所有特殊字符，防止 tiktoken 报错
        self.eos_token = self.enc.encode('<|endoftext|>', allowed_special='all')[0]
        self.pad_token = self.eos_token 
        
        self.data = []
        print("正在解析多轮对话版 DPO 偏好数据集...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    
                    # 1. 拼接多轮历史上下文 (Context)
                    prompt_str = ""
                    for turn in item.get("context", []):
                        role = turn.get("role", "")
                        text = turn.get("text", "").strip()
                        if role == "human" or role == "user":
                            prompt_str += f"User: {text}\n\n"
                        elif role == "assistant":
                            prompt_str += f"Assistant: {text}\n\n"
                            
                    # 2. 加上最后一句等待回答的“暗号”
                    prompt_str += "Assistant: "
                    
                    # 3. 提取 Chosen 和 Rejected
                    chosen_text = item["chosen"]["text"].strip()
                    rejected_text = item["rejected"]["text"].strip()
                    
                    self.data.append({
                        "prompt": prompt_str, 
                        "chosen": chosen_text, 
                        "rejected": rejected_text
                    })
                except Exception as e:
                    # 跳过格式损坏的行
                    continue
                
        print(f"多轮 DPO 数据集加载完成！共成功解析 {len(self.data)} 条对决数据。")

    def __len__(self):
        return len(self.data)
        
    def _pad_and_mask(self, prompt_ids, answer_ids):
        # 拼接并加上结束符
        input_ids = prompt_ids + answer_ids + [self.eos_token]
        # 把问题部分的梯度屏蔽掉 (-100)
        labels = [-100] * len(prompt_ids) + answer_ids + [self.eos_token]
        
        # 截断或填充
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token] * pad_len
            labels = labels + [-100] * pad_len
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt_ids = self.enc.encode(item["prompt"], allowed_special='all')
        chosen_ids = self.enc.encode(item["chosen"], allowed_special='all')
        rejected_ids = self.enc.encode(item["rejected"], allowed_special='all')
        
        # 分别处理好答案和坏答案
        chosen_input_ids, chosen_labels = self._pad_and_mask(prompt_ids, chosen_ids)
        rejected_input_ids, rejected_labels = self._pad_and_mask(prompt_ids, rejected_ids)
        
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels