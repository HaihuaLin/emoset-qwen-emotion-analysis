import os
import re
import glob
import json
import random
import shutil
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from modelscope import snapshot_download
from transformers import AutoProcessor
from peft import PeftModel

# 动态加载最适配多模态视觉任务的模型类
try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelClass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-4B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_LORA_DIR = os.path.join(OUTPUT_DIR, "qwen_lora_emoset")
EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "lora_eval_results.json")
ZERO_SHOT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "zero_shot_eval_results.json")

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

def load_lora_model_and_processor(lora_dir=DEFAULT_LORA_DIR):
    print(f"[1/2] 正在加载基座模型与 LoRA 适配器...")
    print(f"      - 基座路径: {MODELS_DIR}")
    print(f"      - LoRA路径: {lora_dir}")

    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"❌ 未找到 LoRA 权重目录: {lora_dir}！请先运行 python train_lora.py 完成训练。")

    processor = AutoProcessor.from_pretrained(MODELS_DIR, trust_remote_code=True)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    # 1. 加载基座
    base_model = AutoModelClass.from_pretrained(
        MODELS_DIR,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # 2. 挂载 LoRA
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    print(f"[*] LoRA 模型加载完成，运行设备: {model.device}, 数据类型: {torch_dtype}")
    return model, processor

def collect_test_dataset_samples(samples_per_class=100, seed=42):
    """
    收集与训练隔离的测试集（严格保持与 train_lora.py 中划分逻辑一致）
    """
    print(f"[2/2] 正在准备独立测试集（每类固定采样 {samples_per_class} 张，Seed={seed}）...")
    random.seed(seed)
    
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    category_image_map = {cat: [] for cat in EMOTION_CATEGORIES}

    for cat in EMOTION_CATEGORIES:
        for ext in exts:
            pattern = os.path.join(DATA_DIR, "**", cat, ext)
            found = glob.glob(pattern, recursive=True)
            category_image_map[cat].extend(found)
        category_image_map[cat] = sorted(list(set(category_image_map[cat])))

    test_samples = []
    print("\n--- 测试集样本确认 ---")
    for cat in EMOTION_CATEGORIES:
        all_imgs = category_image_map[cat]
        tot = len(all_imgs)
        if tot == 0:
            print(f"  ❌ 类别 [{cat}]: 未找到图片！")
            continue
        
        shuffled = all_imgs.copy()
        random.shuffle(shuffled)
        test_imgs = shuffled[:samples_per_class]
        print(f"  ✓ 类别 [{cat:12s}]: 提取独立测试样本 {len(test_imgs):3d} 张")
        for f in test_imgs:
            test_samples.append({"file_path": f, "ground_truth": cat})
    
    random.shuffle(test_samples)
    print(f"\n[*] 测试集构建完毕！测试样本总计: {len(test_samples)} 张（与零样本基线完全一致）")
    return test_samples

def extract_predicted_emotion(response_text):
    """从模型的输出文本中提取预测的情感类别"""
    # 1. 优先正则匹配结构化标签
    match = re.search(r"(?:最终情感分类|Final Emotion|Emotion|情感类别)[：:\s]*([a-zA-Z\u4e00-\u9fa5]+)", response_text, re.IGNORECASE)
    if match:
        raw_val = match.group(1).strip().lower()
        if raw_val in EMOTION_CATEGORIES:
            return raw_val
        if raw_val in EMOTION_CN_TO_EN:
            return EMOTION_CN_TO_EN[raw_val]

    # 2. 检查结尾词
    tail_text = response_text[-100:].lower()
    for cat in EMOTION_CATEGORIES:
        if cat in tail_text:
            return cat
    for cn, en in EMOTION_CN_TO_EN.items():
        if cn in tail_text:
            return en

    # 3. 全文关键词
    full_text = response_text.lower()
    for cat in EMOTION_CATEGORIES:
        if cat in full_text:
            return cat
    for cn, en in EMOTION_CN_TO_EN.items():
        if cn in full_text:
            return en

    return "unknown"

def run_lora_inference(model, processor, image_path):
    prompt_text = (
        "你是一个顶尖的图像情感分析专家。请仔细观察这张图片，并完成以下任务：\n"
        f"1. 从给定的 8 种离散情感类别中选择最符合该图像的一项：{', '.join(EMOTION_CATEGORIES)}。\n"
        "2. 简要分析画面中的关键视觉线索（如色调/明暗度、人物表情、肢体动作、主体物体及场景环境），说明为什么会引发这种情感。\n\n"
        "【输出规范】请在回答的最后一行务必使用如下格式明确给出结论：\n"
        "【最终情感分类】: "
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
            max_new_tokens=150,
            do_sample=False  # 评测使用确定性贪婪解码
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

def load_zero_shot_baseline():
    """读取零样本评测基线结果（用于对比）"""
    if os.path.exists(ZERO_SHOT_RESULTS_PATH):
        try:
            with open(ZERO_SHOT_RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("summary", {})
        except Exception:
            pass
    # 默认使用之前的实测基准
    return {
        "overall_accuracy": "16.00%",
        "per_class_accuracy": {
            "amusement": "22.00%", "anger": "13.00%", "awe": "38.00%",
            "contentment": "30.00%", "disgust": "24.00%", "excitement": "1.00%",
            "fear": "0.00%", "sadness": "0.00%"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B EmoSet LoRA 微调模型独立评测脚本")
    parser.add_argument("--lora_dir", type=str, default=DEFAULT_LORA_DIR, help="LoRA 适配器目录 (默认 output/qwen_lora_emoset)")
    parser.add_argument("--samples_per_class", type=int, default=100, help="每类测试样本数 (默认 100)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (与零样本评测保持一致)")
    args = parser.parse_args()

    print("=" * 70)
    print("      🎯 Qwen3.5-4B LoRA 微调模型效果评测与前后对比")
    print(f"      LoRA 权重路径: {args.lora_dir}")
    print(f"      独立测试集量: 8 类别 x {args.samples_per_class} 张 = {8 * args.samples_per_class} 张")
    print("=" * 70)

    # 1. 加载模型
    model, processor = load_lora_model_and_processor(args.lora_dir)

    # 2. 收集测试集
    samples = collect_test_dataset_samples(samples_per_class=args.samples_per_class, seed=args.seed)
    if not samples:
        print("❌ 未获取到测试样本！")
        return

    # 3. 推理测试
    total_samples = len(samples)
    total_correct = 0
    class_stats = {cat: {"total": 0, "correct": 0} for cat in EMOTION_CATEGORIES}
    detailed_results = []

    print("\n" + "=" * 70)
    print("                 开始批量 LoRA 模型推理评测...")
    print("=" * 70)

    pbar = tqdm(samples, desc="评测进度", unit="img", dynamic_ncols=True)
    for idx, sample in enumerate(pbar):
        file_path = sample["file_path"]
        gt = sample["ground_truth"]

        pred, full_response = run_lora_inference(model, processor, file_path)
        is_correct = (pred == gt)

        total_correct += int(is_correct)
        class_stats[gt]["total"] += 1
        class_stats[gt]["correct"] += int(is_correct)

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

    # 4. 读取零样本基准进行对比
    zero_shot_summary = load_zero_shot_baseline()
    overall_acc = (total_correct / total_samples) * 100
    zero_overall_str = zero_shot_summary.get("overall_accuracy", "16.00%")
    zero_overall_float = float(zero_overall_str.replace("%", ""))
    gain_float = overall_acc - zero_overall_float

    # 5. 打印对比报告
    print("\n" + "=" * 75)
    print("                     🏆 LoRA 微调 vs 零样本基线 评测对比报告")
    print("=" * 75)
    print(f"{'情感类别 (Category)':<15} | {'零样本基线 (Zero-Shot)':<20} | {'LoRA微调后 (Ours)':<18} | {'绝对提升 (Gain)':<12}")
    print("-" * 75)
    
    per_class_summary = {}
    for cat in EMOTION_CATEGORIES:
        stats = class_stats[cat]
        c_tot = stats["total"]
        c_cor = stats["correct"]
        c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
        per_class_summary[cat] = f"{c_acc:.2f}%"

        zero_class_str = zero_shot_summary.get("per_class_accuracy", {}).get(cat, "0.00%")
        zero_class_float = float(zero_class_str.replace("%", ""))
        diff = c_acc - zero_class_float
        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"

        print(f"{cat:<15} | {zero_class_str:>18} | {c_acc:>16.2f}% | {diff_str:>12}")

    print("-" * 75)
    overall_gain_str = f"+{gain_float:.2f}%" if gain_float >= 0 else f"{gain_float:.2f}%"
    print(f"{'⭐ 总体平均 (Overall)':<15} | {zero_overall_str:>18} | {overall_acc:>16.2f}% | {overall_gain_str:>12}")
    print("=" * 75)

    # 6. 保存详细结果
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "lora_dir": args.lora_dir,
                "total_samples": total_samples,
                "total_correct": total_correct,
                "overall_accuracy": f"{overall_acc:.2f}%",
                "zero_shot_accuracy": zero_overall_str,
                "overall_gain": overall_gain_str,
                "per_class_accuracy": per_class_summary
            },
            "details": detailed_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[*] 详细逐张评测结果已保存至: {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    main()
