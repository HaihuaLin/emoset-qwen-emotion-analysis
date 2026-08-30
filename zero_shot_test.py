import os
import re
import glob
import json
import random
import shutil
import zipfile
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from modelscope import snapshot_download
from transformers import AutoProcessor

# 动态加载最适配多模态视觉任务的模型类
try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelClass

# 获取项目根目录路径 (/mnt/workspace/emoset-qwen-emotion-analysis)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-4B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "zero_shot_eval_results.json")

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

# 中英文映射，便于解析
EMOTION_CN_TO_EN = {
    "娱乐": "amusement", "愉悦": "amusement", "开心": "amusement", "搞笑": "amusement",
    "愤怒": "anger", "生气": "anger", "暴怒": "anger",
    "敬畏": "awe", "震撼": "awe", "惊叹": "awe",
    "满足": "contentment", "安详": "contentment", "惬意": "contentment", "舒适": "contentment",
    "厌恶": "disgust", "恶心": "disgust", "反感": "disgust",
    "兴奋": "excitement", "激动": "excitement", "狂喜": "excitement",
    "恐惧": "fear", "害怕": "fear", "惊恐": "fear",
    "悲伤": "sadness", "难过": "sadness", "伤心": "sadness", "忧郁": "sadness"
}

def clean_stale_locks():
    """清理因之前 Ctrl+C 中断残留的 ModelScope 文件锁，防止死锁等待"""
    lock_dir = os.path.expanduser("~/.cache/modelscope/.lock")
    if os.path.exists(lock_dir):
        try:
            shutil.rmtree(lock_dir, ignore_errors=True)
            print("[*] 已自动清理历史残留的文件锁 (.lock)...")
        except Exception:
            pass

def load_model_and_processor(model_id="Qwen/Qwen3.5-4B"):
    print(f"[1/3] 正在加载模型（目标路径: {MODELS_DIR}）...")
    clean_stale_locks()
    
    if os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0:
        print(f"[*] 发现项目本地已存在的模型文件: {MODELS_DIR}")
        model_dir = MODELS_DIR
    else:
        cache_base = os.path.expanduser("~/.cache/modelscope/models/Qwen--Qwen3.5-4B")
        migrated = False
        if os.path.exists(cache_base):
            snapshots = glob.glob(os.path.join(cache_base, "snapshots", "*"))
            if snapshots and os.path.exists(snapshots[0]):
                cached_path = snapshots[0]
                print(f"[*] 发现系统缓存中的模型，正在自动转移至: {MODELS_DIR} (释放系统盘空间)...")
                os.makedirs(os.path.dirname(MODELS_DIR), exist_ok=True)
                shutil.move(cached_path, MODELS_DIR)
                try:
                    shutil.rmtree(cache_base, ignore_errors=True)
                except Exception:
                    pass
                model_dir = MODELS_DIR
                migrated = True
        
        if not migrated:
            print(f"[*] 正在从 ModelScope 下载模型至项目目录: {MODELS_DIR} ...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            try:
                model_dir = snapshot_download(model_id, local_dir=MODELS_DIR)
            except Exception:
                model_dir = snapshot_download(model_id, cache_dir=os.path.join(PROJECT_ROOT, "models"))
    
    print(f"[*] 模型路径确认: {model_dir}")
    print(f"[2/3] 正在加载 Processor 与 {AutoModelClass.__name__} 进入显卡 (A10)...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelClass.from_pretrained(
        model_dir,
        device_map=device_map,
        dtype=torch_dtype,
        trust_remote_code=True
    )
    model.eval()
    print(f"[*] 模型加载完成，运行设备: {model.device}, 数据类型: {torch_dtype}")
    return model, processor

def collect_balanced_dataset_samples(samples_per_class=100, seed=42):
    """从 EmoSet 的 8 个情感类别中各采样指定数量的图像"""
    print(f"[3/3] 正在准备平衡评估数据集（每个类别采样 {samples_per_class} 张）...")
    random.seed(seed)
    
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    category_image_map = {cat: [] for cat in EMOTION_CATEGORIES}

    # 扫描已解压的数据目录
    for cat in EMOTION_CATEGORIES:
        for ext in exts:
            # 匹配类似 data/EmoSet/extracted/**/sadness/*.jpg 或 data/EmoSet/**/sadness/*.jpg
            pattern = os.path.join(DATA_DIR, "**", cat, ext)
            found = glob.glob(pattern, recursive=True)
            category_image_map[cat].extend(found)
        # 去重
        category_image_map[cat] = list(set(category_image_map[cat]))

    # 汇总并采样
    balanced_samples = []
    print("\n--- 各类别样本发现统计 ---")
    for cat in EMOTION_CATEGORIES:
        available_files = category_image_map[cat]
        count = len(available_files)
        sample_count = min(samples_per_class, count)
        if count == 0:
            print(f"  ❌ 类别 [{cat}]: 未找到图片！请确认数据解压完整性。")
            continue
        
        sampled_files = random.sample(available_files, sample_count)
        print(f"  ✓ 类别 [{cat:12s}]: 发现 {count:6d} 张图片，采样 {sample_count:3d} 张")
        for f in sampled_files:
            balanced_samples.append({
                "file_path": f,
                "ground_truth": cat
            })
    
    # 随机打乱测试顺序
    random.shuffle(balanced_samples)
    print(f"\n[*] 采样完成！评估样本总计: {len(balanced_samples)} 张图片（预期: {len(EMOTION_CATEGORIES) * samples_per_class} 张）")
    return balanced_samples

def extract_predicted_emotion(response_text):
    """从模型的输出文本中提取预测的情感类别"""
    # 1. 优先匹配结构化格式：【最终情感分类】: category 或 Final Emotion: category
    match = re.search(r"(?:最终情感分类|Final Emotion|Emotion|情感类别)[：:\s]*([a-zA-Z\u4e00-\u9fa5]+)", response_text, re.IGNORECASE)
    if match:
        raw_val = match.group(1).strip().lower()
        if raw_val in EMOTION_CATEGORIES:
            return raw_val
        if raw_val in EMOTION_CN_TO_EN:
            return EMOTION_CN_TO_EN[raw_val]

    # 2. 检查文本最后 150 个字符中出现的关键词（结论通常在结尾）
    tail_text = response_text[-150:].lower()
    for cat in EMOTION_CATEGORIES:
        if cat in tail_text:
            return cat
    for cn, en in EMOTION_CN_TO_EN.items():
        if cn in tail_text:
            return en

    # 3. 全文关键词出现频率统计
    full_text = response_text.lower()
    for cat in EMOTION_CATEGORIES:
        if cat in full_text:
            return cat
    for cn, en in EMOTION_CN_TO_EN.items():
        if cn in full_text:
            return en

    return "unknown"

def run_zero_shot_inference(model, processor, image_path):
    prompt_text = (
        "你是一个顶尖的图像情感分析专家。请仔细观察这张图片，并完成以下任务：\n"
        f"1. 从给定的 8 种离散情感类别中选择最符合该图像的一项：{', '.join(EMOTION_CATEGORIES)}。\n"
        "2. 简要分析画面中的关键视觉线索（如：色调/明暗度、人物面部表情、肢体动作、主体物体及场景环境），说明为什么会引发这种情感。\n\n"
        "【输出规范】请在回答的最后一行务必使用如下格式明确给出结论：\n"
        "【最终情感分类】: <8种英文类别之一>"
    )

    image = Image.open(image_path).convert("RGB")
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
            max_new_tokens=300,
            do_sample=False  # 评测使用确定性贪婪解码，保证结果一致性与极高速度
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    predicted_emotion = extract_predicted_emotion(response)
    return predicted_emotion, response

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B EmoSet 零样本批量评测脚本")
    parser.add_argument("--samples_per_class", type=int, default=100, help="每个情感类别采样的图片数量（默认 100）")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子")
    args = parser.parse_args()

    print("=" * 65)
    print("   Qwen3.5-4B 零样本（Zero-shot）情感分析评测与准确率统计")
    print(f"   类别总数: 8 | 每类采样: {args.samples_per_class} | 总测试量: {8 * args.samples_per_class}")
    print("=" * 65)

    # 1. 加载模型
    model, processor = load_model_and_processor("Qwen/Qwen3.5-4B")

    # 2. 准备每类 100 张样本
    samples = collect_balanced_dataset_samples(samples_per_class=args.samples_per_class, seed=args.seed)
    if not samples:
        print("❌ 未获取到有效测试样本，请检查数据集路径！")
        return

    # 3. 统计计数器初始化
    total_samples = len(samples)
    total_correct = 0
    class_stats = {cat: {"total": 0, "correct": 0} for cat in EMOTION_CATEGORIES}
    detailed_results = []

    print("\n" + "=" * 65)
    print("                 开始批量零样本推理评测...")
    print("=" * 65)

    # 4. 使用 tqdm 带实时进度条与实时准确率展示
    pbar = tqdm(samples, desc="评测进度", unit="img", dynamic_ncols=True)
    for idx, sample in enumerate(pbar):
        file_path = sample["file_path"]
        gt = sample["ground_truth"]

        pred, full_response = run_zero_shot_inference(model, processor, file_path)
        is_correct = (pred == gt)

        # 更新计数
        total_correct += int(is_correct)
        class_stats[gt]["total"] += 1
        class_stats[gt]["correct"] += int(is_correct)

        # 实时动态更新进度条后缀
        current_acc = (total_correct / (idx + 1)) * 100
        pbar.set_postfix({
            "总体Acc": f"{current_acc:.2f}%",
            "当前": f"{gt[:3]}->{pred[:3]}",
            "正确数": f"{total_correct}/{idx+1}"
        })

        detailed_results.append({
            "index": idx + 1,
            "file_path": file_path,
            "ground_truth": gt,
            "predicted": pred,
            "is_correct": is_correct,
            "full_response": full_response
        })

    # 5. 打印最终全面评测报告
    overall_acc = (total_correct / total_samples) * 100
    print("\n" + "=" * 65)
    print("                     🏆 零样本评测结果报告")
    print("=" * 65)
    print(f"{'情感类别 (Category)':<18} | {'测试样本数':<10} | {'预测正确数':<10} | {'准确率 (Accuracy)':<15}")
    print("-" * 65)
    for cat in EMOTION_CATEGORIES:
        stats = class_stats[cat]
        c_tot = stats["total"]
        c_cor = stats["correct"]
        c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
        print(f"{cat:<18} | {c_tot:<10} | {c_cor:<10} | {c_acc:>6.2f}%")
    
    print("-" * 65)
    print(f"{'⭐ 总体平均 (Overall)':<18} | {total_samples:<10} | {total_correct:<10} | {overall_acc:>6.2f}%")
    print("=" * 65)

    # 6. 保存详细结果到 JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": total_samples,
                "total_correct": total_correct,
                "overall_accuracy": f"{overall_acc:.2f}%",
                "per_class_accuracy": {
                    cat: f"{(class_stats[cat]['correct'] / class_stats[cat]['total'] * 100):.2f}%"
                    for cat in EMOTION_CATEGORIES if class_stats[cat]['total'] > 0
                }
            },
            "details": detailed_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[*] 详细逐张推理结果已保存至: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
