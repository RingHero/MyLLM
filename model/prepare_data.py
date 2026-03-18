import json
import numpy as np
import tiktoken
import os
from tqdm import tqdm

#用于转换json文件为Bin文件

def prepare_data(input_path, output_path):
    enc = tiktoken.get_encoding('gpt2')
    eos_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
    
    # 我们以追加模式写入二进制文件
    # 注意：这里使用 uint16 就可以存下 GPT-2 的 50257 词表 (0~65535)
    # 这能帮你省下一半的硬盘空间！
    dtype = np.uint16 
    
    # 每次积攒一批数据再写入硬盘，提高速度
    buffer = []
    buffer_size = 1000000  # 100万个token写一次
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # 假设文件很大，我们逐行读取
        for line in tqdm(f, desc="Tokenizing"):
            try:
                text = json.loads(line.strip())['text']
                encoded_text = enc.encode(text)
                buffer.extend(encoded_text + [eos_token])
                
                # 缓冲区满了就刷入硬盘
                if len(buffer) >= buffer_size:
                    # 转换成 numpy 数组然后写入
                    arr = np.array(buffer, dtype=dtype)
                    with open(output_path, 'ab') as out_f:
                        out_f.write(arr.tobytes())
                    buffer = [] # 清空缓冲区
            except Exception as e:
                continue
                
        # 写入最后剩下的
        if len(buffer) > 0:
            arr = np.array(buffer, dtype=dtype)
            with open(output_path, 'ab') as out_f:
                out_f.write(arr.tobytes())

    print(f"数据预处理完成！保存在 {output_path}")

if __name__ == '__main__':
    prepare_data(
        '/home/apulis-dev/userdata/mydata/mobvoi_seq_monkey_general_open_corpus.jsonl',
        '/home/apulis-dev/userdata/mydata/train_data.bin'
    )
