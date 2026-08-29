import os
import torch
from PIL import Image
from modelscope import snapshot_download, MsDataset
from transformers import AutoModelForCausalLM, AutoProcessor

# EmoSet 定义的 8 种基础离散情感类别
EMOTION_CATEGORIES = [
    "amusement",    # 娱乐/愉悦
    "anger",        # 愤怒
    "awe",          # 敬畏
    "contentment",  # 满足
    "disgust",      # 厌恶
    "excitement",   # 兴奋
    "fear",         # 恐惧
    "sadness"       # 悲伤
]

def load_model_and_processor(model_id="Qwen/Qwen3.5-4B"):
    print(f"[1/4] 正在检查或从 ModelScope 下载模型: {model_id} ...")
    model_dir = snapshot_download(model_id)
    print(f"模型下载/加载路径: {model_dir}")

    print("[2/4] 正在加载 Processor 与 Model...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # 自动检测 GPU/CPU 与合适的数据类型
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.eval()
    print(f"模型已就绪，运行设备: {model.device}, 数据类型: {torch_dtype}")
    return model, processor

def load_dataset_samples(dataset_id="weisir001/EmoSet", num_samples=3):
    print(f"[3/4] 正在从 ModelScope 加载数据集: {dataset_id} ...")
    # 尝试加载测试集或训练集
    try:
        ds = MsDataset.load(dataset_id, split='test')
    except Exception:
        print("未找到 test 分割，尝试加载 train 分割...")
        ds = MsDataset.load(dataset_id, split='train')
    
    print(f"数据集加载成功，样本总数: {len(ds)}")
    return ds

def run_zero_shot_inference(model, processor, image, ground_truth=None):
    prompt_text = (
        "你是一个顶尖的图像情感分析专家。请仔细观察这张图片，并完成以下任务：\n"
        f"1. 从给定的 8 种离散情感类别中选择最符合该图像的一项：{', '.join(EMOTION_CATEGORIES)}。\n"
        "2. 简要分析画面中的关键视觉线索（如：色调/明暗度、人物面部表情、肢体动作、主体物体及场景环境），说明为什么会引发这种情感。\n"
        "请使用规范的格式输出最终的情感类别与分析理由。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # 应用聊天模板
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 抽取并处理图像与文本输入
    inputs = processor(
        text=[text], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8
        )
    
    # 仅截取新生成的 tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return response

def main():
    print("=" * 60)
    print("      Qwen3.5-4B 零样本（Zero-shot）图像情感分析测试")
    print("=" * 60)

    # 1. 加载模型
    model, processor = load_model_and_processor("Qwen/Qwen3.5-4B")

    # 2. 加载数据集
    dataset = load_dataset_samples("weisir001/EmoSet", num_samples=3)

    print("\n[4/4] 开始进行零样本推理测试...")
    for idx, sample in enumerate(dataset):
        if idx >= 3:  # 默认测试前 3 张样例
            break

        print("\n" + "-" * 50)
        print(f"【样本 #{idx + 1}】")

        # 获取图像对象
        image = None
        for img_key in ['image', 'img', 'Image', 'file', 'image_path']:
            if img_key in sample:
                val = sample[img_key]
                if isinstance(val, Image.Image):
                    image = val
                elif isinstance(val, str) and os.path.exists(val):
                    image = Image.open(val).convert("RGB")
                break
        
        if image is None:
            print(f"样本 {idx + 1} 中未解析到有效图像，字段包含: {list(sample.keys())}")
            continue

        # 获取真实标签（如果存在）
        gt_label = sample.get('label') or sample.get('emotion') or sample.get('category') or "未知"
        print(f"真实标注情感 (Ground Truth): {gt_label}")

        # 模型推理
        print("正在进行 Qwen3.5-4B 推理中...")
        prediction = run_zero_shot_inference(model, processor, image, ground_truth=gt_label)

        print(f"\n模型预测与分析结果:\n{prediction}")
        print("-" * 50)

    print("\n测试流程已全部完成！")

if __name__ == "__main__":
    main()
