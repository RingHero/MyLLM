import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model_lm import LMmodel, LMConfig
from model.mydataset import MyDataset

def train():
    # 从环境变量获取分布式信息
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    # 1. 模型配置与初始化
    lmconfig = LMConfig()
    model = LMmodel(lmconfig).to(local_rank)

    # 数据集准备
    dataset = MyDataset('/home/apulis-dev/userdata/mydata/train_data.bin', max_length=lmconfig.max_len)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_subset, batch_size=6, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=6, sampler=val_sampler, num_workers=2, pin_memory=True)
    
    # 2. 优化器、调度器初始化
    max_lr = 2e-4 
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    MAX_TRAIN_STEPS = 240000 
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=MAX_TRAIN_STEPS,
        pct_start=0.05, 
        anneal_strategy='cos'
    )
    
    save_step = 10000 
    global_step = 0
    start_epoch = 0

    # 断点续训功能
    resume_checkpoint_path = '/home/apulis-dev/userdata/tmp/model_step_120000.pt' 
    
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        if rank == 0:
            print(f"🔄 发现存档点 {resume_checkpoint_path}，正在恢复训练状态...")
        
        # 先加载到 CPU 防爆显存
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        
        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复调度器状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            if rank == 0:
                print("⚠️ 警告：旧存档未保存 scheduler 状态，学习率将按新的 MAX_TRAIN_STEPS 重新计算！")
                
        # 恢复全局步数
        global_step = checkpoint['step']
        
        start_epoch = global_step // len(train_loader)
        
        del checkpoint
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"✅ 成功从 Step {global_step} 恢复训练！目标步数：{MAX_TRAIN_STEPS}")

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. 训练循环 (从 start_epoch 开始) 
    # 其实epoch已经不重要了，预训练的数据很大而且卡比较少的话，跑一个月可能一轮epoch没结束呢
    # 所以主要看step,根据step数来评估模型，保存参数
    for epoch in range(start_epoch, 101):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(local_rank), y.to(local_rank)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0 and rank == 0:
                print(f"Epoch {epoch} | Step {global_step}/{MAX_TRAIN_STEPS} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if global_step >= MAX_TRAIN_STEPS:
                if rank == 0:
                    print(f"✅ 达到最大训练步数 {MAX_TRAIN_STEPS}，预训练圆满结束！")
                dist.destroy_process_group()
                return 

            # 按步数进行验证和保存
            if global_step % save_step == 0:
                model.eval()
                val_loss = 0.0
                eval_batches = 100 
                
                with torch.no_grad():
                    for i, (eval_x, eval_y) in enumerate(val_loader):
                        if i >= eval_batches: 
                            break
                        eval_x, eval_y = eval_x.to(local_rank), eval_y.to(local_rank)
                        logits = model(eval_x)
                        loss = criterion(logits.view(-1, logits.size(-1)), eval_y.view(-1))
                        val_loss += loss.item()

                val_loss_tensor = torch.tensor(val_loss).to(local_rank)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / world_size / eval_batches

                if rank == 0:
                    print(f"--- Step {global_step} 验证完成, Val Loss: {avg_val_loss:.4f} ---")
                    checkpoint = {
                        'step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': avg_val_loss,
                    }
                    torch.save(checkpoint, f'/home/apulis-dev/userdata/tmp/model_step_{global_step}.pt')
                
                model.train() 

    dist.destroy_process_group()

if __name__ == "__main__":
    train()