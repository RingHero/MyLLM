import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from model.model_lm import LMmodel, LMConfig
from model.mydataset import MyDataset

lmconfig = LMConfig()
model = LMmodel(lmconfig)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数设置 ignore_index=-100，与数据集中填充部分的标签对应
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# 训练数据
train_dataset = MyDataset('/home/apulis-dev/userdata/mydata/mobvoi_seq_monkey_general_open_corpus.jsonl', max_length=lmconfig.max_len)

# 分割训练集和验证集
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # [B, T, vocab_size]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss

if __name__ == '__main__':
    for epoch in range(5):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')