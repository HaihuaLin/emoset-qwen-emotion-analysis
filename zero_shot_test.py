import os
import glob
import torch
from PIL import Image
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

# 获取项目根目录路径 (/mnt/workspace/emoset-qwen-emotion-analysis)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-4B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")

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
    print(f"[1/4] 正在加载模型（目标项目路径: {MODELS_DIR}）...")
    
    # 优先从项目本地 models/ 目录加载，如果不存在则自动下载到该目录
    if os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0:
        print(f"[*] 发现项目本地已存在的模型文件: {MODELS_DIR}")
        model_dir = MODELS_DIR
    else:
        # 如果 /root/.cache 中有之前下好的，也可以直接利用，或指定 local_dir 下载到项目目录
        try:
            print(f"[*] 正在将模型下载并存储至项目目录: {MODELS_DIR} ...")
            model_dir = snapshot_download(model_id, local_dir=MODELS_DIR)
        except Exception:
            model_dir = snapshot_download(model_id, cache_dir=os.path.join(PROJECT_ROOT, "models"))
    
    print(f"模型加载就绪，路径: {model_dir}")

    print("[2/4] 正在加载 Processor 与 Model 进入显卡 (A10)...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
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

def load_dataset_samples(num_samples=3):
    print(f"[3/4] 正在检查项目数据目录: {DATA_DIR} ...")
    samples = []

    # 扫描项目本地 data/EmoSet 目录下的图像
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    if os.path.exists(DATA_DIR):
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    # 如果本地还没有下载数据集，提示或尝试下载
    if not image_paths:
        print(f"[*] 项目目录 {DATA_DIR} 暂无图片，正在自动调用下载与解压...")
        from download_dataset import download_and_extract_dataset
        download_and_extract_dataset("weisir001/EmoSet")
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    if not image_paths:
        raise FileNotFoundError(f"未能在 {DATA_DIR} 中找到图片！请先运行 python download_dataset.py 下载数据。")

    print(f"[*] 在项目数据目录中发现 {len(image_paths)} 张图片，挑选前 {num_samples} 张进行测试。")
    for p in image_paths[:num_samples]:
        parent_dir = os.path.basename(os.path.dirname(p))
        inferred_label = parent_dir if parent_dir in EMOTION_CATEGORIES else "未知"
        samples.append({
            'image': Image.open(p).convert("RGB"),
            'label': inferred_label,
            'file_path': p
        })

    return samples

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

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
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

    # 1. 加载模型（存储在项目目录 models/ 下）
    model, processor = load_model_and_processor("Qwen/Qwen3.5-4B")

    # 2. 加载数据集（存储在项目目录 data/ 下）
    samples = load_dataset_samples(num_samples=3)

    print("\n[4/4] 开始进行零样本推理测试...")
    for idx, sample in enumerate(samples):
        print("\n" + "-" * 50)
        print(f"【样本 #{idx + 1}】")

        image = sample['image']
        gt_label = sample['label']
        print(f"图片路径: {sample['file_path']}")
        print(f"真实标注情感 (Ground Truth): {gt_label}")

        print("正在进行 Qwen3.5-4B 推理中...")
        prediction = run_zero_shot_inference(model, processor, image, ground_truth=gt_label)

        print(f"\n模型预测与分析结果:\n{prediction}")
        print("-" * 50)

    print("\n✅ 测试流程已全部完成！")

if __name__ == "__main__":
    main()
