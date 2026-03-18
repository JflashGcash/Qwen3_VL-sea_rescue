"""
Qwen3-VL-8B + LoRA best_model 微调效果测试
用与零样本测试完全相同的5张图片，方便对比
"""
import json
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch

MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
LORA_PATH = '/root/autodl-tmp/qwen_vl/lora_output/best_model'
BASE = '/root/autodl-tmp/qwen_vl/finetune_data'

print('=' * 60)
print('加载 Qwen3-VL-8B + LoRA best_model')
print('=' * 60)

print('\n[1/3] 加载基座模型（4bit量化）...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
print('  基座模型加载完成！')

print('\n[2/3] 加载 LoRA 权重...')
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print(f'  LoRA 权重已加载：{LORA_PATH}')

print('\n[3/3] 加载处理器...')
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print('  处理器加载完成！')

with open(os.path.join(BASE, 'val.json'), 'r') as f:
    val_data = json.load(f)

tested_images = set()
test_samples = []
for sample in val_data:
    img_path = sample['messages'][0]['content'][0]['image']
    if img_path not in tested_images and len(test_samples) < 5:
        tested_images.add(img_path)
        test_samples.append(sample)

print('\n' + '=' * 60)
print('开始测试（共5张图片）')
print('=' * 60)

for i, sample in enumerate(test_samples):
    img_path_raw = sample['messages'][0]['content'][0]['image']
    img_path_full = os.path.join(BASE, img_path_raw)
    ground_truth = sample['messages'][1]['content'][0]['text']

    print(f'\n{"=" * 60}')
    print(f'测试 {i+1}/5：{img_path_raw}')
    print(f'{"=" * 60}')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path_full}"},
                {"type": "text", "text": "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print(f'\n【LoRA微调后回答】：')
    print(response)
    print(f'\n【正确答案】：')
    print(ground_truth)
    print()

print('=' * 60)
print('best_model 测试完成！')
print('请与 test_zero_shot.py 的结果对比。')
print('=' * 60)