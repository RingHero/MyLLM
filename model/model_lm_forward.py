#-------------------------------------------
# 带KVCahce的模型
#-------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken


class LMConfig:
    def __init__(self, args=None):
        if args is not None:
            self.model_name = args.model_name
            self.model_path = args.model_path
            self.hidden_size = args.hidden_size
            self.num_layers = args.num_layers
            self.num_heads = args.num_heads
            self.dropout = args.dropout
            self.max_len = args.max_len
            self.vocab_size = args.vocab_size
            self.pad_token_id = args.pad_token_id
            self.cls_token_id = args.cls_token_id
            self.sep_token_id = args.sep_token_id
            self.unk_token_id = args.unk_token_id
        else:
            self.num_layers = 12
            self.hidden_size = 768
            self.num_heads = 12
            self.dropout = 0.1
            self.max_len = 1024
            self.vocab_size = 50257
            self.rms_norm_eps = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    
def precompute_freqs_cis(dim: int, max_len: int, theta: float = 1e5):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
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

    q_embed = (q * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))
    return q_embed, k_embed

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

    def forward(self, x, freqs_cis, attn_mask=None, kv_cache=None):
        bsz, seq_len, _ = x.shape
        xq, xk ,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #多头注意力
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)    #[B, nhead, seq_len, head_dim]
        xk = xk.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        xv = xv.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        #RoPE编码，旋转变换
        fcos, fsin = freqs_cis
        #print(fcos.shape, fsin.shape)
        xq, xk = apply_rotary_pos_emb(xq, xk, fcos, fsin)
        
        #KVcache实现
        if kv_cache is not None:
            past_k, past_v = kv_cache
            xk = torch.cat([past_k, xk], dim=2)#注意在seq_len维度进行拼接
            xv = torch.cat([past_v, xv], dim=2)
        #缓存当前的KV
        new_kv_cache = (xk, xv)

        score = (xq @ xk.transpose(-1,-2)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            score = score.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = (attn_weights @ xv)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, new_kv_cache

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


class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.preLN = RMSNorm(config.hidden_size)
        self.self_attn = Attention(config)
        self.ffn = FeedForward(config)
        self.postLN = RMSNorm(config.hidden_size)

    def forward(self, x, freqs_cis, attn_mask=None, kv_cache=None):
        attn_out, new_kv_cache = self.self_attn(self.preLN(x), freqs_cis, attn_mask, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.postLN(x))
        return x, new_kv_cache
    
class LMmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        attn_mask = torch.tril(torch.ones(self.config.max_len,self.config.max_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.config.hidden_size // self.config.num_heads, self.config.max_len)

        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        x = self.tok_embeddings(input_ids)
        bsz, seq_len, _ = x.shape
        
        #kv_cache & start_pos
        start_pos = 0
        if past_key_values is not None:
            start_pos = past_key_values[0][0].shape[2]
        
        freqs_cis = (
            self.freqs_cos[start_pos : start_pos + seq_len], 
            self.freqs_sin[start_pos : start_pos + seq_len]
        )
        
        mask = None
        if seq_len > 1:
            mask = self.attn_mask[:, :, :seq_len, :seq_len]
            
        new_key_values = []    
        for i, layer in enumerate(self.layers):
            #提出当前层的kv_cache
            layer_cache = past_key_values[i] if past_key_values is not None else None#如果是第一层，则为None
            #前向传播
            x, new_layer_cache = layer(x, freqs_cis, mask, layer_cache)
            #合并多层的kv_cache
            if use_cache:
                new_key_values.append(new_layer_cache)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_key_values
        else:
            return logits


def load_model_and_tokenizer(model_path=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    enc = tiktoken.get_encoding("gpt2")
    config = LMConfig()
    model = LMmodel(config)
    
    if model_path:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
    
    model = model.to(device)  # 将模型移动到目标设备
    model.eval()
    return model, enc, device  # 返回 device 方便后续使用


def generate(
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

    #初始化kv_cache
    past_key_values = None
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 如果是第一次 (Prefill)，送入全部 input_ids
            # 如果不是第一次 (Decode)，只送入上一步刚刚生成的 1 个最新 Token！
            if past_key_values is None:
                current_input = input_ids
            else:
                current_input = input_ids[:, -1:] # 切片，保持 [1, 1] 形状
            # 拿到输出的同时，拿到更新后的 Cache
            logits, past_key_values = model(current_input, past_key_values=past_key_values, use_cache=True)  # 模型已在目标设备，输入也在同一设备
            print("past_key_values:", past_key_values[0][0].shape)
        next_token_logits = logits[0, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == enc.eot_token:
            break
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        

    generated_ids = input_ids[0].tolist()
    generated_text = enc.decode(generated_ids)
    return generated_text


if __name__ == '__main__':
    model, enc, eos_id = load_model_and_tokenizer(model_path='D:/BaiduNetdiskDownload/sft_model_step_5493.pt')  # eos_id 这里其实没用
    prompt = '我是'
    output = generate(model, enc, prompt, max_new_tokens=1000, temperature=0.9)
    print(enc.encode(output))
    print("生成结果：", output)


