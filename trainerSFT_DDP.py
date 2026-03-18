import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model_lm import LMmodel, LMConfig
from model.SFTdataset import SFTDataset_csv  # 确保这里用的是处理 -100 掩码的 SFTDataset

def train():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    # 1. 模型初始化
    lmconfig = LMConfig()
    model = LMmodel(lmconfig).to(local_rank)

    if rank == 0:
        print("🧠 正在注入预训练基座模型...")
    checkpoint = torch.load('/home/apulis-dev/userdata/tmp/model_step_230000.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    torch.cuda.empty_cache()

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2. 数据集加载
    dataset = SFTDataset_csv('/home/apulis-dev/userdata/mydata/alpaca_data.csv', max_length=lmconfig.max_len)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_subset, batch_size=6, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=6, sampler=val_sampler, num_workers=2, pin_memory=True)
    
    # 学习率调小，防止破坏基座常识
    max_lr = 5e-5 
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    # 设定 Epoch 数，自动计算总步数
    NUM_EPOCHS = 3
    MAX_TRAIN_STEPS = NUM_EPOCHS * len(train_loader)
    
    if rank == 0:
        print(f"📊 SFT 总步数预估: {MAX_TRAIN_STEPS} 步 (共 {NUM_EPOCHS} Epochs)")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=MAX_TRAIN_STEPS,
        pct_start=0.05, 
        anneal_strategy='cos'
    )
    
    # 保存频率调快，每 500 步存一次
    save_step = 500 
    global_step = 0

    # 3. 训练循环 (全新的开始，从 0 跑到 NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(local_rank), y.to(local_rank)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            if global_step % 50 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step}/{MAX_TRAIN_STEPS} | SFT Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            # 验证与保存
            if global_step % save_step == 0 or global_step == MAX_TRAIN_STEPS:
                model.eval()
                val_loss = 0.0
                eval_batches = 50 # SFT 验证集抽测 50 个 batch 足够了
                
                with torch.no_grad():
                    for i, (eval_x, eval_y) in enumerate(val_loader):
                        if i >= eval_batches: 
                            break
                        eval_x, eval_y = eval_x.to(local_rank), eval_y.to(local_rank)
                        logits = model(eval_x)
                        eval_loss = criterion(logits.view(-1, logits.size(-1)), eval_y.view(-1))
                        val_loss += eval_loss.item()

                val_loss_tensor = torch.tensor(val_loss).to(local_rank)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / world_size / eval_batches

                if rank == 0:
                    print(f"--- Step {global_step} 验证完成, SFT Val Loss: {avg_val_loss:.4f} ---")
                    checkpoint = {
                        'step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        # SFT 跑完直接用来推理，通常不需要再存 optimizer 和 scheduler 了，省点空间
                    }
                    torch.save(checkpoint, f'/home/apulis-dev/userdata/tmp/sft_model_step_{global_step}.pt')
                
                model.train() 

    dist.destroy_process_group()

if __name__ == "__main__":
    train()