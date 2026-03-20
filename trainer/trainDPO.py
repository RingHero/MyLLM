import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from model.model_lm import LMmodel, LMConfig
from model.DPOdataset import DPODataset

def get_batch_logps(logits, labels, ignore_index=-100):
    """
    这是一个极其核心的黑魔法函数：计算模型对于特定回答的 "综合对数概率 (Log Probability)"。
    """
    # 1. 错位预测（Next Token Prediction 标配）
    logits = logits[:, :-1, :]  # 拿出除了最后一个词之外的所有预测
    labels = labels[:, 1:]      # 拿出除了第一个词之外的所有目标
    
    # 2. 计算每个词的 log softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 3. 提取出真实 Label 对应的那个词的概率
    # 因为 label 里有 -100，gather 时会报错，所以先把它替换成 0（反正是要被屏蔽的）
    loss_mask = labels != ignore_index
    labels_for_gather = labels.clone()
    labels_for_gather[labels == ignore_index] = 0
    
    # 获取目标词的概率
    per_token_logps = torch.gather(log_probs, dim=2, index=labels_for_gather.unsqueeze(2)).squeeze(2)
    
    # 4. 只把回答部分（mask 为 True）的概率加起来！屏蔽掉 prompt 和 padding
    return (per_token_logps * loss_mask).sum(-1)

def dpo_loss(policy_chosen_logits, policy_rejected_logits, 
             ref_chosen_logits, ref_rejected_logits, 
             chosen_labels, rejected_labels, beta=0.1):
    
    # 1. 计算策略模型 (选手) 对两个答案的打分
    policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)
    policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)
    
    # 2. 计算参考模型 (裁判) 对两个答案的打分
    with torch.no_grad(): # 裁判绝对不参与梯度更新！
        ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
        ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)
        
    # 3. 核心 DPO 公式：计算概率的比值差！
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    # 4. 用 logsigmoid 算出最终 Loss
    loss = -F.logsigmoid(beta * logits).mean()
    
    return loss


def train():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    lmconfig = LMConfig()
    
    # ==========================================
    # 🏟️ 组建 DPO 双子星模型阵列
    # ==========================================
    if rank == 0:
        print("🤖 正在请出 Policy 模型 (选手) 和 Reference 模型 (裁判)...")
    
    # 1. 实例化两个完全一样的模型
    policy_model = LMmodel(lmconfig).to(local_rank)
    ref_model = LMmodel(lmconfig).to(local_rank)
    
    # 2. 给它们加载一模一样的 SFT 权重！
    # 假设这是你用 Alpaca 训出来并且可以正常聊天的那个权重
    sft_ckpt = torch.load('/home/apulis-dev/userdata/tmp/sft_model_step_xxx.pt', map_location='cpu')
    policy_model.load_state_dict(sft_ckpt['model_state_dict'])
    ref_model.load_state_dict(sft_ckpt['model_state_dict'])
    
    del sft_ckpt
    torch.cuda.empty_cache()

    # 3. 🚨 极其关键：彻底冻结裁判！
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # 4. 把选手送进 DDP
    policy_model = DDP(policy_model, device_ids=[local_rank])
    
    # ==========================================
    # 📊 加载数据和优化器
    # ==========================================
    dataset = DPODataset('toy_dpo.jsonl', max_length=lmconfig.max_len)
    # 因为是 toy 数据，就不划分验证集了，全塞进 train
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # 注意：BS 可以设小一点，因为要同时前向传播 4 次！
    train_loader = DataLoader(dataset, batch_size=2, sampler=train_sampler, num_workers=2) 
    
    # DPO 的学习率必须非常小，比 SFT 还要小！
    max_lr = 1e-5 
    optimizer = optim.AdamW(policy_model.parameters(), lr=max_lr, weight_decay=0.1)

    # ... [Scheduler 设定和 SFT 类似] ...
    
    # ==========================================
    # ⚔️ DPO 激战循环
    # ==========================================
    NUM_EPOCHS = 3
    MAX_TRAIN_STEPS = NUM_EPOCHS * len(train_loader)
    global_step = 0
    save_step = 1000
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        policy_model.train()
        
        # DataLoader 现在会吐出 4 个张量！
        for batch_idx, (c_x, c_y, r_x, r_y) in enumerate(train_loader):
            c_x, c_y = c_x.to(local_rank), c_y.to(local_rank) # Chosen 的输入和标签
            r_x, r_y = r_x.to(local_rank), r_y.to(local_rank) # Rejected 的输入和标签
            
            optimizer.zero_grad()
            
            # ==========================================
            # 🌟 核心魔法：拼接 (Concat Trick)
            # ==========================================
            # 把好答案和坏答案在 Batch 维度 (dim=0) 上拼成一个大 Batch
            x_concat = torch.cat([c_x, r_x], dim=0)
            
            # 1. 选手作答 (只进行 1 次前向传播！)
            policy_logits_concat = policy_model(x_concat)
            # 算完之后，用 chunk 一刀劈成两半，恢复成 c 和 r
            policy_c_logits, policy_r_logits = policy_logits_concat.chunk(2, dim=0)
            
            # 2. 裁判作答 (同样拼接起来跑 1 次)
            with torch.no_grad():
                ref_logits_concat = ref_model(x_concat)
                ref_c_logits, ref_r_logits = ref_logits_concat.chunk(2, dim=0)
                
            # 3. 祭出 DPO 神级 Loss！
            loss = dpo_loss(
                policy_c_logits, policy_r_logits,
                ref_c_logits, ref_r_logits,
                c_y, r_y, beta=0.1
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # scheduler.step()
            global_step += 1
            if global_step % 50 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step}/{MAX_TRAIN_STEPS} | SFT Loss: {loss.item():.4f}")
            
            if global_step % save_step == 0 or global_step == MAX_TRAIN_STEPS:
            
                if rank == 0:
                    print(f"Epoch {epoch} | Step {batch_idx} | DPO Loss: {loss.item():.4f}")
                    # ==========================================
                    # 最后保存 policy_model.module.state_dict() 即可！
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': policy_model.module.state_dict(),
                    }
                    torch.save(checkpoint, f'/home/apulis-dev/userdata/tmp/dpo_model_step_{global_step}.pt')
    dist.destroy_process_group()

if __name__ == '__main__':
    train()