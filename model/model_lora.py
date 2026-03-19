import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConfig:
    def __init__(self, args=None):
        if args is not None:
            self.in_dim = args.in_dim
            self.out_dim = args.out_dim
            self.rank = args.rank
        else:
            self.in_dim = 768
            self.out_dim = 768
            self.rank = 16


class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super(LoRA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
    
def apply_lora(model, rank=4):
    for name, module in model.named_modules():
        # 只作用于方阵的LoRA, 其实也就是Attention层的QKV矩阵
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[1], module.weight.shape[0], rank) # 注意权重的形状是反过来的
            lora.to(module.weight.device)
            setattr(module, 'lora', lora)
            origin_forward = module.forward

            def new_forward(x):
                return origin_forward(x) + lora(x)
            
            module.forward = new_forward

