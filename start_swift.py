import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift import SftArguments, sft_main

def main():
    args = SftArguments(
        model='qwen/Qwen3-VL-8B-Instruct',
        train_type='lora',
        dataset=['modelscope/coco_2014_caption:mini'],
        max_length=1024,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        eval_steps=50,
        save_total_limit=2,
        system='You are a helpful assistant.'
    )
    sft_main(args)

if __name__ == '__main__':
    main()
