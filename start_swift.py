import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import SftArguments, sft_main

def main():
    args = SftArguments(
        # 1. 模型与数据
        model='qwen/Qwen3-VL-8B-Instruct',
        dataset=['modelscope/coco_2014_caption:mini'],
        system='You are a helpful assistant.',
        max_length=1024,

        # 2. 训练类型 (关键：你的版本里叫 tuner_type)
        tuner_type='lora',

        # 3. 训练超参数 (必须使用你列表里的全称)
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, # 显存足够可以加这个提高稳定性
        eval_steps=50,
        save_total_limit=2,
        bf16=True, # A6000 支持 bf16，建议开启加速

        # 4. LoRA 具体配置 (可选，如果不写会用默认值)
        lora_rank=8,
        lora_alpha=32
    )
    sft_main(args)

if __name__ == '__main__':
    main()
