import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import SftArguments, sft_main

args = SftArguments(
    model='qwen/Qwen3-VL-8B-Instruct',
    train_type='lora',  # 注意：新版可能叫 train_type 或 sft_type
    dataset=['modelscope/coco_2014_caption:mini'],
    max_length=2048,
    learning_rate=1e-4,
    batch_size=1,
    eval_steps=50,
    save_total_limit=2,
    system='You are a helpful assistant.'
)

if __name__ == '__main__':
    sft_main(args)
