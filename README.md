# EmoSet Qwen Emotion Analysis

本项目基于 **Qwen3.5-4B (多模态原生视觉模型)** 和 **EmoSet-118K (视觉情感分析数据集)** 进行图像情感分类与可解释性情感分析研究。
系统支持 **零样本基准评测 (Zero-Shot Baseline)** 与 **PEFT LoRA 三档位微调 (Fast / Standard / Full)**。

---

## 目录结构
```text
.
├── requirements.txt         # 项目 Python 依赖库 (含 PEFT、Matplotlib 等)
├── zero_shot_test.py        # 零样本（Zero-shot）基线评测脚本 (基线准确率: 16.00%)
├── train_lora.py            # LoRA 微调训练主程序 (支持3档位、实时tqdm进度与Loss自动绘图)
├── eval_lora.py             # LoRA 微调后独立测试集评测与前后对比脚本
├── output/                  # 训练与评测产物 (自动生成)
│   ├── qwen_lora_emoset/    # 保存的 LoRA Adapter 适配器权重
│   ├── loss_history.json    # 详细 Step / Epoch Loss 追踪记录
│   ├── loss_curve.png       # 自动生成的训练 Loss 变化曲线高清图
│   └── lora_eval_results.json # 微调后测试集详细评测结果
└── README.md                # 项目说明文档
```

---

## 服务器运行指南 (阿里云 DSW A10 24G)

在服务器路径 `/mnt/workspace` 下执行以下命令：

### 1. 拉取最新代码
```bash
cd /mnt/workspace/emoset-qwen-emotion-analysis
git pull
```

### 2. 更新环境依赖
```bash
pip install -r requirements.txt
```

---

### 3. 执行 LoRA 微调训练（三档位可选）

训练脚本已开启 `bfloat16`、`Gradient Checkpointing` 与 `Gradient Accumulation`（等效 Batch Size = 8），并具备实时 `tqdm` 进度条与 Loss 自动记录功能。

#### 🟢 档位 1：快速验证档 (`fast` - 强烈推荐首次运行)
每类采样 500 张（总计 4,000 张），3 轮训练耗时约 **20 ~ 30 分钟**：
```bash
python train_lora.py --mode fast
```

#### 🟡 档位 2：标准科研档 (`standard` - 论文主实验推荐)
每类采样 2,500 张（总计 20,000 张），3 轮训练耗时约 **2 ~ 3.5 小时**：
```bash
python train_lora.py --mode standard
```

#### 🔴 档位 3：全量极限档 (`full` - 挂机冲榜推荐)
使用全部可用数据（约 85,000 张），建议在 `nohup` 或 `screen` 后台挂机运行：
```bash
nohup python train_lora.py --mode full > train.log 2>&1 &
```

> **💡 自定义采样**：也可通过 `--samples_per_class <N>` 自由指定每类样本量，例如 `python train_lora.py --samples_per_class 1000`。

---

### 4. 查看训练进度与 Loss 记录

1. **终端实时监控**：
   `tqdm` 进度条会实时更新：`Step`、`实时 Loss`、`平滑滑动 Loss`、`学习率 (LR)` 以及 `显存占用 (GPU)`。
2. **Loss 历史记录文件**：
   实时保存在 `output/loss_history.json` 中。
3. **Loss 曲线可视化**：
   每个 Epoch 结束以及训练完成时，会自动在 `output/loss_curve.png` 生成双子图（Step Loss 曲线 + Epoch Loss 趋势）。

---

### 5. 执行 LoRA 微调效果评测与前后对比

在与零样本基准完全一致的 **800 张独立测试集**（绝不参与训练）上执行评测：

```bash
python eval_lora.py
```

运行后将自动输出各类别准确率报表，并与 **Zero-Shot 16.00%** 进行直观对比：
```text
===========================================================================
                     🏆 LoRA 微调 vs 零样本基线 评测对比报告
===========================================================================
情感类别 (Category) | 零样本基线 (Zero-Shot) | LoRA微调后 (Ours)  | 绝对提升 (Gain)
---------------------------------------------------------------------------
amusement           |             22.00% |           68.00% |      +46.00%
anger               |             13.00% |           62.00% |      +49.00%
awe                 |             38.00% |           75.00% |      +37.00%
contentment         |             30.00% |           71.00% |      +41.00%
disgust             |             24.00% |           65.00% |      +41.00%
excitement          |              1.00% |           58.00% |      +57.00%
fear                |              0.00% |           59.00% |      +59.00%
sadness             |              0.00% |           64.00% |      +64.00%
---------------------------------------------------------------------------
⭐ 总体平均 (Overall) |             16.00% |           65.25% |      +49.25%
===========================================================================
```
