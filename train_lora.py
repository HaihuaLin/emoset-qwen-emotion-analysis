import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import sys
import glob
import json
import time
import random
import shutil
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 无头绘图后端，支持服务器无界面运行
import matplotlib.pyplot as plt

from modelscope import snapshot_download
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 动态加载最适配多模态视觉任务的模型类
try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelClass

# 项目路径定义
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-4B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EmoSet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LORA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "qwen_lora_emoset")
LOSS_HISTORY_PATH = os.path.join(OUTPUT_DIR, "loss_history.json")
LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
TRAIN_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "train_summary.json")

# EmoSet 8 种基础离散情感类别
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

# 统一提示词模板
PROMPT_INSTRUCTION = (
    "你是一个顶尖的图像情感分析专家。请仔细观察这张图片，并完成以下任务：\n"
    f"1. 从给定的 8 种离散情感类别中选择最符合该图像的一项：{', '.join(EMOTION_CATEGORIES)}。\n"
    "2. 简要分析画面中的关键视觉线索（如色调/明暗度、人物表情、肢体动作、主体物体及场景环境），说明为什么会引发这种情感。\n\n"
    "【输出规范】请在回答的最后一行务必使用如下格式明确给出结论：\n"
    "【最终情感分类】: "
)

def clean_stale_locks():
    """清理因之前 Ctrl+C 中断残留的 ModelScope 文件锁"""
    lock_dir = os.path.expanduser("~/.cache/modelscope/.lock")
    if os.path.exists(lock_dir):
        try:
            shutil.rmtree(lock_dir, ignore_errors=True)
            print("[*] 已自动清理历史残留的文件锁 (.lock)...")
        except Exception:
            pass

def load_model_and_processor(model_id="Qwen/Qwen3.5-4B", precision="4bit", gradient_checkpointing=True):
    print(f"[1/4] 正在加载基座模型（目标路径: {MODELS_DIR}）...")
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
    print(f"[2/4] 正在加载 Processor 与 {AutoModelClass.__name__} (精度模式: {precision})...")
    
    # 限制单图 visual tokens 数量为 ~200-300，彻底消除异常大图显存暴涨
    min_pixels = 256 * 28 * 28
    max_pixels = 384 * 28 * 28
    try:
        processor = AutoProcessor.from_pretrained(
            model_dir, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels, 
            trust_remote_code=True
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    # 4-bit QLoRA 量化配置 (NF4 格式，显存暴降至 2.4G)
    if precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelClass.from_pretrained(
            model_dir,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        print(f"[*] 已成功启用 4-bit NF4 (QLoRA) 极限显存量化！基座权重占用仅 ~2.5 GB！")
    elif precision == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelClass.from_pretrained(
            model_dir,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        print(f"[*] 已成功启用 8-bit 量化！基座权重占用 ~4.5 GB！")
    else:
        model = AutoModelClass.from_pretrained(
            model_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"[*] 模型加载完成，运行设备: {model.device}, 运行精度: {precision}")
    return model, processor, model_dir

def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    print(f"[3/4] 正在注入 LoRA 适配层 (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model

def prepare_dataset_splits(mode="fast", custom_samples_per_class=None, seed=42):
    """
    划分并准备数据集：
    1. 固定隔离 800 张平衡测试集（每类 100 张，固定 seed=42），绝不参与训练；
    2. 根据 mode 或 custom_samples_per_class 从剩余样本中抽取训练集与验证集。
    """
    print(f"\n[4/4] 正在扫描 EmoSet 数据集并构建三档位划分 (Mode: {mode}, Seed: {seed})...")
    random.seed(seed)
    
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    category_image_map = {cat: [] for cat in EMOTION_CATEGORIES}

    for cat in EMOTION_CATEGORIES:
        for ext in exts:
            pattern = os.path.join(DATA_DIR, "**", cat, ext)
            found = glob.glob(pattern, recursive=True)
            category_image_map[cat].extend(found)
        category_image_map[cat] = sorted(list(set(category_image_map[cat])))

    # 1. 严格隔离 800 张独立测试集 (每类 100 张)
    test_set_per_class = 100
    test_samples = []
    train_pool_map = {}

    print("\n--- 数据集分布与测试集隔离统计 ---")
    for cat in EMOTION_CATEGORIES:
        all_imgs = category_image_map[cat]
        tot = len(all_imgs)
        if tot == 0:
            print(f"  ❌ 类别 [{cat}]: 未找到图片！请确认数据目录。")
            continue
        
        # 固定打乱并抽取前 100 张作为测试集
        shuffled = all_imgs.copy()
        random.shuffle(shuffled)
        test_imgs = shuffled[:test_set_per_class]
        remaining_pool = shuffled[test_set_per_class:]
        
        for f in test_imgs:
            test_samples.append({"file_path": f, "label": cat})
        train_pool_map[cat] = remaining_pool
        print(f"  ✓ 类别 [{cat:12s}]: 发现总数 {tot:5d} 张 | 隔离测试集 {len(test_imgs):3d} 张 | 可用训练池 {len(remaining_pool):5d} 张")

    # 2. 根据档位确定每类采样数量
    if custom_samples_per_class is not None and custom_samples_per_class > 0:
        samples_target = custom_samples_per_class
        mode_desc = f"自定义采样 ({samples_target} 张/类)"
    elif mode == "fast":
        samples_target = 500
        mode_desc = "快速验证档 (500 张/类，总训练集约 4,000 张)"
    elif mode == "standard":
        samples_target = 2500
        mode_desc = "标准科研档 (2,500 张/类，总训练集约 20,000 张)"
    elif mode == "full":
        samples_target = None  # 使用全部剩余样本
        mode_desc = "全量极限档 (使用全部剩余可用样本，约 80,000~95,000 张)"
    else:
        raise ValueError(f"未知档位模式: {mode}")

    print(f"\n>>> 当前训练档位配置: 【{mode_desc}】")

    train_samples = []
    val_samples = []
    val_ratio = 0.05  # 5% 样本作为验证集评估泛化 loss

    for cat in EMOTION_CATEGORIES:
        pool = train_pool_map.get(cat, [])
        if not pool:
            continue
        
        if samples_target is not None:
            actual_sample_cnt = min(samples_target, len(pool))
            sampled = pool[:actual_sample_cnt]
        else:
            sampled = pool
        
        val_cnt = max(10, int(len(sampled) * val_ratio))
        val_imgs = sampled[:val_cnt]
        train_imgs = sampled[val_cnt:]

        for f in train_imgs:
            train_samples.append({"file_path": f, "label": cat})
        for f in val_imgs:
            val_samples.append({"file_path": f, "label": cat})

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"[*] 数据集划分完成！")
    print(f"    - 训练集 (Train): {len(train_samples):6d} 张")
    print(f"    - 验证集 (Val)  : {len(val_samples):6d} 张")
    print(f"    - 隔离测试集(Test): {len(test_samples):6d} 张")

    return train_samples, val_samples, test_samples

class EmoSetMultiModalDataset(Dataset):
    """EmoSet 多模态数据集类"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample["file_path"]
        label = sample["label"]

        try:
            image = Image.open(file_path).convert("RGB")
            # 限制图像最大边长为 448，防止超高分辨率图片产生数千个 visual token 导致显存暴涨 OOM
            image.thumbnail((448, 448), Image.Resampling.BILINEAR)
        except Exception:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_INSTRUCTION}
        ]
        assistant_content = [
            {"type": "text", "text": f"{label}"}
        ]

        user_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        return {
            "image": image,
            "user_messages": user_messages,
            "full_messages": full_messages,
            "label": label,
            "file_path": file_path
        }

class EmoSetDataCollator:
    """多模态专用 Batch 整理器，使用官方 Processor 原生处理变长视觉张量与动态分辨率"""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        user_messages_list = [item["user_messages"] for item in batch]
        full_messages_list = [item["full_messages"] for item in batch]

        # 1. 构造 prompt 文本与完整对话文本
        prompt_texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in user_messages_list
        ]
        full_texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in full_messages_list
        ]

        # 2. 官方 Processor 统一进行批处理与动态视觉 Patch 拼接
        batch_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # 3. 逐样本精确计算 Prompt Token 长度并构建 Labels 掩码 (-100 过滤)
        labels = batch_inputs.input_ids.clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        for i, (p_text, img) in enumerate(zip(prompt_texts, images)):
            single_p_inputs = self.processor(
                text=[p_text],
                images=[img],
                return_tensors="pt"
            )
            p_len = single_p_inputs.input_ids.shape[1]
            # 将 Prompt 与图像占位部分掩码为 -100
            labels[i, :p_len] = -100

        # 将 Batch Padding 区域也掩码为 -100
        labels[batch_inputs.input_ids == pad_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs

def plot_and_save_loss_curve(loss_history, output_path=LOSS_CURVE_PATH):
    """自动绘制训练 Loss 曲线与 Epoch 变化趋势"""
    if not loss_history.get("steps"):
        return

    steps_data = loss_history["steps"]
    epochs_data = loss_history.get("epochs", [])

    steps = [d["step"] for d in steps_data]
    losses = [d["loss"] for d in steps_data]
    smooth_losses = [d["smooth_loss"] for d in steps_data]

    plt.figure(figsize=(12, 5), dpi=150)

    # 子图 1: 逐 Step 训练损失与平滑损失
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, color='#94a3b8', alpha=0.4, label='Raw Step Loss')
    plt.plot(steps, smooth_losses, color='#2563eb', linewidth=2.0, label='Smoothed Loss (EMA)')
    plt.title('Training Step Loss Curve', fontsize=13, fontweight='bold')
    plt.xlabel('Global Steps', fontsize=11)
    plt.ylabel('CrossEntropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    # 子图 2: 逐 Epoch 训练与验证损失趋势
    plt.subplot(1, 2, 2)
    if epochs_data:
        ep_list = [d["epoch"] for d in epochs_data]
        train_ep_loss = [d["train_loss"] for d in epochs_data]
        plt.plot(ep_list, train_ep_loss, marker='o', color='#16a34a', linewidth=2.0, label='Train Epoch Loss')
        if any(d.get("val_loss") is not None for d in epochs_data):
            val_ep_loss = [d["val_loss"] for d in epochs_data if d.get("val_loss") is not None]
            plt.plot(ep_list[:len(val_ep_loss)], val_ep_loss, marker='s', color='#dc2626', linewidth=2.0, label='Val Epoch Loss')
        plt.title('Epoch-Level Loss Trend', fontsize=13, fontweight='bold')
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Loss', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
    else:
        plt.text(0.5, 0.5, 'Epoch Loss will appear after 1st epoch', horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def evaluate_validation_loss(model, val_loader, device):
    """计算验证集 Loss"""
    model.eval()
    total_val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is not None:
                total_val_loss += loss.item()
                val_steps += 1
    model.train()
    return (total_val_loss / val_steps) if val_steps > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B EmoSet LoRA 三档位微调系统")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "standard", "full"], 
                        help="训练档位选择: fast(快速验证, ~4k张), standard(标准科研, ~20k张), full(全量极限, ~85k张)")
    parser.add_argument("--precision", type=str, default="4bit", choices=["4bit", "8bit", "bf16", "fp16"], 
                        help="微调精度模式: 4bit(默认QLoRA，显存仅需~6G), 8bit(~10G), bf16(~18G), fp16")
    parser.add_argument("--samples_per_class", type=int, default=None, 
                        help="自定义每类训练样本数 (覆盖 mode 的默认采样量)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数 (默认 3)")
    parser.add_argument("--batch_size", type=int, default=1, help="单卡单步 Batch Size (推荐 1，极度稳定省显存)")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数 (等效 Batch Size = batch_size * grad_accum = 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率 (默认 1e-4)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank 维度 (默认 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha 系数 (默认 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout (默认 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("      🚀 Qwen3.5-4B EmoSet 多模态 LoRA 情感分析微调系统")
    print(f"      档位模式: {args.mode} | 运行精度: {args.precision} | 轮数: {args.epochs} | 等效 BatchSize: {args.batch_size * args.grad_accum}")
    print("=" * 70)

    # 1. 准备数据集划分
    train_samples, val_samples, test_samples = prepare_dataset_splits(
        mode=args.mode,
        custom_samples_per_class=args.samples_per_class,
        seed=args.seed
    )

    # 2. 加载基座模型与 Processor (支持 4-bit QLoRA 极限显存节省)
    model, processor, _ = load_model_and_processor(precision=args.precision, gradient_checkpointing=True)

    # 3. 注入 LoRA
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # 4. 构建 DataLoader
    train_dataset = EmoSetMultiModalDataset(train_samples)
    val_dataset = EmoSetMultiModalDataset(val_samples) if val_samples else None
    data_collator = EmoSetDataCollator(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0
    ) if val_dataset else None

    # 5. 优化器与学习率调度器 (优先采用 PagedAdamW8bit 防显存尖峰)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = max(10, int(total_steps * 0.05))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("[*] 成功启用 8-bit PagedAdamW 优化器，彻底消除优化器显存尖峰！")
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("[*] 启用标准 AdamW 优化器")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print("\n--- 训练超参数汇总 ---")
    print(f"  - 运行精度模式: {args.precision}")
    print(f"  - 训练集样本数: {len(train_dataset):d}")
    print(f"  - 验证集样本数: {len(val_dataset) if val_dataset else 0:d}")
    print(f"  - 总迭代步数 (Global Steps): {total_steps:d}")
    print(f"  - 预热步数 (Warmup Steps) : {warmup_steps:d}")
    print(f"  - 初始学习率: {args.lr}")
    print(f"  - 权重保存路径: {LORA_OUTPUT_DIR}")

    # 6. 初始化 Loss 记录器
    loss_history = {
        "metadata": {
            "mode": args.mode,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "steps": [],
        "epochs": []
    }

    device = model.device
    model.train()
    global_step = 0
    smooth_loss = None
    alpha_smooth = 0.95
    best_val_loss = float("inf")
    start_train_time = time.time()

    print("\n" + "=" * 70)
    print("                  🔥 开始执行 LoRA 微调训练...")
    print("=" * 70)

    # 7. 训练主循环
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        step_in_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                outputs = model(**inputs)
                loss = outputs.loss

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss_val = loss.item()
                epoch_train_loss += loss_val
                step_in_epoch += 1

                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()
            except torch.cuda.OutOfMemoryError:
                print("\n[Warning] 捕获到单个异常大图引发的 OOM，已自动清理显存碎片并跳过，保证训练平稳推进！")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            if smooth_loss is None:
                smooth_loss = loss_val
            else:
                smooth_loss = alpha_smooth * smooth_loss + (1 - alpha_smooth) * loss_val

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 定期释放显存碎片
                if global_step % 20 == 0:
                    torch.cuda.empty_cache()

                current_lr = lr_scheduler.get_last_lr()[0]
                step_record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss_val, 4),
                    "smooth_loss": round(smooth_loss, 4),
                    "lr": float(f"{current_lr:.2e}"),
                    "timestamp": round(time.time() - start_train_time, 2)
                }
                loss_history["steps"].append(step_record)

                if global_step % 10 == 0:
                    with open(LOSS_HISTORY_PATH, "w", encoding="utf-8") as f:
                        json.dump(loss_history, f, indent=2, ensure_ascii=False)

            mem_info = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_info = f"{allocated:.1f}G"

            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "Step": f"{global_step}/{total_steps}",
                "Loss": f"{loss_val:.4f}",
                "AvgLoss": f"{smooth_loss:.4f}",
                "LR": f"{current_lr:.2e}",
                "GPU": mem_info
            })

        avg_train_loss = epoch_train_loss / max(1, step_in_epoch)
        val_loss = None
        if val_loader:
            print(f"\n[*] 正在计算 Epoch {epoch} 验证集 Loss...")
            val_loss = evaluate_validation_loss(model, val_loader, device)
            print(f"    ✓ Epoch {epoch} 验证集 Loss: {val_loss:.4f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "time_seconds": round(time.time() - epoch_start_time, 2)
        }
        loss_history["epochs"].append(epoch_record)

        with open(LOSS_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)
        plot_and_save_loss_curve(loss_history, LOSS_CURVE_PATH)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(OUTPUT_DIR, "qwen_lora_emoset_best")
            print(f"[*] 发现更佳验证集 Loss ({val_loss:.4f})，保存最佳权重至: {best_dir}")
            model.save_pretrained(best_dir)

    print("\n" + "=" * 70)
    print("                  🎉 LoRA 微调训练圆满完成！")
    print("=" * 70)
    print(f"[*] 正在保存最终 LoRA 适配器权重至: {LORA_OUTPUT_DIR} ...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    processor.save_pretrained(LORA_OUTPUT_DIR)

    total_duration = time.time() - start_train_time
    hours, rem = divmod(total_duration, 3600)
    mins, secs = divmod(rem, 60)

    summary = {
        "status": "success",
        "training_mode": args.mode,
        "epochs_completed": args.epochs,
        "total_steps": global_step,
        "final_smooth_loss": round(smooth_loss, 4) if smooth_loss else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
        "total_time": f"{int(hours)}h {int(mins)}m {int(secs)}s",
        "lora_output_dir": LORA_OUTPUT_DIR,
        "loss_history_path": LOSS_HISTORY_PATH,
        "loss_curve_path": LOSS_CURVE_PATH
    }

    with open(TRAIN_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[*] 训练总用时: {summary['total_time']}")
    print(f"[*] 最终平滑 Loss: {summary['final_smooth_loss']}")
    print(f"[*] Loss 历史记录文件: {LOSS_HISTORY_PATH}")
    print(f"[*] Loss 曲线图导出至: {LOSS_CURVE_PATH}")
    print(f"[*] 可直接运行 python eval_lora.py 进行独立测试集效果评测！")

if __name__ == "__main__":
    main()
