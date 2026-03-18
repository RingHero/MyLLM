import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken


# 定义模型配置类
# 应该要导入一个parse_args，但是我没做，所以这里直接写死
class LMConfig:
    def __init__(self, args=None):
        if args is not None:
            self.hidden_size = args.hidden_size
            self.num_layers = args.num_layers
            self.num_heads = args.num_heads
            self.dropout = args.dropout
            self.max_len = args.max_len
            self.vocab_size = args.vocab_size
            self.rms_norm_eps = args.rms_norm_eps
        else:
            self.num_layers = 12
            self.hidden_size = 768
            self.num_heads = 12
            self.dropout = 0.1
            self.max_len = 512
            self.vocab_size = 50257
            self.rms_norm_eps = 1e-5


#RMSNorm的实现
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 如果需要转换精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

#RoPE编码基底设置
def precompute_freqs_cis(dim: int, max_len: int, theta: float = 1e5):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))   #公式: 1 / (theta ** (2i / dim))
    t = torch.arange(max_len, dtype=torch.float32)  #公式: t 此处可以不用设置device，因为它在一开始只声明一次随后不与模型参数交互
    freqs = torch.outer(t, freqs)   #外积
    freqs_cos = torch.cos(freqs).repeat(1, 2)
    freqs_sin = torch.sin(freqs).repeat(1, 2)
    return (freqs_cos, freqs_sin)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # x: (..., dim)  将最后一维视为连续的二维对
        x = x.reshape(*x.shape[:-1], -1, 2)          # (..., dim//2, 2)
        x1, x2 = x.unbind(dim=-1)                    # 各 (..., dim//2)
        x_rot = torch.stack([-x2, x1], dim=-1)        # (..., dim//2, 2)
        return x_rot.flatten(-2)                      # (..., dim)

    q_embed = (q * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))    #旋转公式：q * cos + q_rot * sin
    k_embed = (k * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))
    return q_embed, k_embed

#注意力机制
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
    
        #self.register_buffer('attn_mask', torch.tril(torch.ones((config.max_len, config.max_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0))

    def forward(self, x, freqs_cis, attn_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk ,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #多头注意力
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)    #[B, nhead, seq_len, head_dim]
        xk = xk.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        xv = xv.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # 应用RoPE编码，旋转变换
        fcos, fsin = freqs_cis
        #print(fcos.shape, fsin.shape)
        xq, xk = apply_rotary_pos_emb(xq, xk, fcos, fsin)

        score = (xq @ xk.transpose(-1,-2)) / math.sqrt(self.head_dim)
        #print(self.attn_mask.shape)
        # 注意力掩码
        if attn_mask is not None:
            score = score.masked_fill(attn_mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        #print(score.shape)
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = (attn_weights @ xv)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

# 线性层带门控机制
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.down_proj = nn.Linear(4 * self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x)))

#Decoder_only
class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.preLN = RMSNorm(config.hidden_size)
        self.self_attn = Attention(config)
        self.ffn = FeedForward(config)
        self.postLN = RMSNorm(config.hidden_size)

    def forward(self, x, freqs_cis, attn_mask=None):
        x = x + self.self_attn(self.preLN(x), freqs_cis, attn_mask)
        x = x + self.ffn(self.postLN(x))
        return x
    
class LMmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #定义注意力掩码矩阵
        attn_mask = torch.tril(torch.ones(self.config.max_len,self.config.max_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        #RoPE编码
        freqs_cos, freqs_sin = precompute_freqs_cis(self.config.hidden_size // self.config.num_heads, self.config.max_len)


        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 掩码和编码都只需要在初始化时计算一次，后续不需要再计算了，通过model.to(device)移动到目标设备
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        # 词向量初始化
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) 
        # lm_head生成logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)
        bsz, seq_len, _ = x.shape
        freqs_cis = (self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])
        for layer in self.layers:
            x = layer(x, freqs_cis, self.attn_mask[:,:,:seq_len,:seq_len])
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def load_model_and_tokenizer(model_path=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    enc = tiktoken.get_encoding("gpt2") #采用gpt2的tokenizer
    config = LMConfig()
    model = LMmodel(config)
    
    if model_path:  #如果提供了模型权重，加载进去
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model = model.to(device)  # 将模型移动到目标设备
    model.eval()
    return model, enc, device  # 返回 device 方便后续使用


def generate(#生成文本，此处实现无KVCahce，非常慢
    model,
    enc,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True,
    device=None  # 新增 device 参数
):
    if device is None:
        device = next(model.parameters()).device  # 自动从模型参数获取设备
    
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # 直接在目标设备创建

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)  
            print("logits:", logits.shape)
        next_token_logits = logits[0, -1, :]

        # temperature策略
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == enc.eot_token:
            break

    generated_ids = input_ids[0].tolist()
    generated_text = enc.decode(generated_ids)
    return generated_text


if __name__ == '__main__':
    model, enc, eos_id = load_model_and_tokenizer(model_path=None)
    prompt = '今天天气真'
    output = generate(model, enc, prompt, max_new_tokens=30, temperature=1.0)
    print(enc.encode(output))
    print("生成结果：", output)

