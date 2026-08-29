import os
import glob
import torch
from PIL import Image
from modelscope import snapshot_download
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
    samples = []

    # 方案 1：优先尝试通过 MsDataset 加载
    try:
        from modelscope.msdatasets import MsDataset
        print("尝试通过 MsDataset 加载数据集...")
        try:
            ds = MsDataset.load(dataset_id, split='test')
        except Exception:
            ds = MsDataset.load(dataset_id, split='train')
        
        for idx, item in enumerate(ds):
            if idx >= num_samples:
                break
            samples.append(item)
        if len(samples) > 0:
            print(f"MsDataset 加载成功，提取到 {len(samples)} 个测试样本。")
            return samples
    except Exception as e:
        print(f"MsDataset 加载跳过 ({e})，切换为本地 snapshot_download 模式下载数据集...")

    # 方案 2：如果 MsDataset 依赖缺失，使用 dataset_snapshot_download 直接下载文件
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
        dataset_dir = dataset_snapshot_download(dataset_id)
    except Exception:
        dataset_dir = snapshot_download(dataset_id, repo_type='dataset')
    
    print(f"数据集已下载至本地目录: {dataset_dir}")
    
    # 扫描目录下的图像文件
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    
    if not image_paths:
        raise FileNotFoundError(f"在数据集目录 {dataset_dir} 中未找到可用图片！")

    print(f"在数据集目录中发现 {len(image_paths)} 张图片，挑选前 {num_samples} 张进行测试。")
    for p in image_paths[:num_samples]:
        # 从文件夹名称或路径中推测情感标签（如 /amusement/001.jpg）
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
    samples = load_dataset_samples("weisir001/EmoSet", num_samples=3)

    print("\n[4/4] 开始进行零样本推理测试...")
    for idx, sample in enumerate(samples):
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
        if 'file_path' in sample:
            print(f"文件路径: {sample['file_path']}")
        print(f"真实标注情感 (Ground Truth): {gt_label}")

        # 模型推理
        print("正在进行 Qwen3.5-4B 推理中...")
        prediction = run_zero_shot_inference(model, processor, image, ground_truth=gt_label)

        print(f"\n模型预测与分析结果:\n{prediction}")
        print("-" * 50)

    print("\n测试流程已全部完成！")

if __name__ == "__main__":
    main()
