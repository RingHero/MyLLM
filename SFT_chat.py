import torch
import torch.nn.functional as F
from .model.model_lm_forward import LMmodel, LMConfig
import tiktoken

#-----------------------------------------
# 用于测试SFT后的模型效果，此时模型才能像人一样跟你对话
#-----------------------------------------


def load_sft_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"正在加载 SFT 模型权重: {model_path} ...")
    enc = tiktoken.get_encoding("gpt2")
    config = LMConfig()
    model = LMmodel(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model, enc, device

def chat_generate(
    model,
    enc,
    user_input,
    max_new_tokens=200,
    temperature=0.3,      
    top_k=50,
    repetition_penalty=1.1,
    device=None
):
    if device is None:
        device = next(model.parameters()).device

    prompt = f"User: {user_input}\n\nAssistant: "
    
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    past_key_values = None
    
    # 记录只属于生成的回答部分
    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                current_input = input_ids
            else:
                current_input = input_ids[:, -1:]
                
            logits, past_key_values = model(current_input, past_key_values=past_key_values, use_cache=True)
            
        next_token_logits = logits[0, -1, :]

        # 重复惩罚
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids): # 惩罚刚生成的词
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        # 温度控制
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Top-K 采样
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # 采样得到新词
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        
        if next_token.item() == enc.eot_token:
            break

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        generated_ids.append(next_token.item())

    # 只解码并返回“Assistant:”后面新生成的内容
    return enc.decode(generated_ids)

if __name__ == '__main__':
    model_path = 'D:/BaiduNetdiskDownload/sft_model_step_30000.pt' 
    model, enc, device = load_sft_model(model_path)
    
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("助手: ", end="", flush=True)
        
        answer = chat_generate(model, enc, user_input, max_new_tokens=1000, temperature=0.5)
        print(answer)
        print("-" * 50)