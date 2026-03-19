import torch
import torch.nn.functional as F
from model.model_lm_forward import LMmodel, LMConfig
from model.model_lora import apply_lora, LoRAConfig
import tiktoken



def load_model_lora(model_path, lora_path, device):
    print(f"🧠正在加载基座模型...")
    enc = tiktoken.get_encoding("gpt2")
    config = LMConfig()
    model = LMmodel(config)

    #加载ckpt
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #加载lora
    print(f"🧠正在加载LoRA模型...")
    lora_config = LoRAConfig()
    apply_lora(model, lora_config.rank)
    lora_checkpoint = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_checkpoint, strict=False)#这里得要加个False， 不然torch找不到lora的参数

    model = model.to(device)
    model.eval()

    return model, enc


def lora_generate(
    model,
    enc,
    user_input,#prompt
    max_new_tokens=200,
    temperature=0.3,
    top_k=50,
    repetition_penalty=1.1,#重复惩罚系数
    device=None        
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prompt = f"User: {user_input}\n\Assistant: "

    input_ids = enc.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    #KVcache
    past_key_values = None

    #记录生成部分的token
    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                current_input = model(input_ids)
            else:
                current_input = model(input_ids, past_key_values=past_key_values, use_cache=True)
            # current_input : [B, seq_len, vocab_size]
            next_token_logits = current_input[0, -1, :]
        
        #重复惩罚
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty
        
        #温度采样：
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


    return enc.decode(generated_ids)


if __name__ == "__main__":
    model_path = 'D:/BaiduNetdiskDownload/model_step_230000.pt'
    lora_path = 'D:/BaiduNetdiskDownload/sft_lora_step_25000.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, enc = load_model_lora(model_path, lora_path, 'cpu')
    print("☁模型加载完成，开始聊天")
    print("-" * 50)
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("助手: ", end="", flush=True)
        
        answer = lora_generate(model, enc, user_input, max_new_tokens=100, temperature=0.5)
        print(answer)
        print("-" * 50)

