# 基于 DAB-Deformable-DETR 的 SAR 舰船识别系统创新与改进方案

## 1. 目标与设计原则
本次改进围绕 HRSID 小目标舰船检测场景，遵循以下原则：
- 每个创新点都提供 `main.py` 启停接口，便于做可重复消融实验。
- 默认配置不改变原始基线行为（默认均为关闭）。
- 实现尽量模块化，避免对原训练主干造成侵入式改写。

---

## 2. 已实现创新点

## 2.1 创新点 A：EMA 教师模型（训练稳定性 + 泛化）
### 动机
SAR 数据集规模相对有限，训练后期参数抖动容易导致验证集性能不稳定。  
EMA（Exponential Moving Average）可对模型权重做时间平滑，通常能提升验证集 mAP 稳定性。

### 实现位置
- `main.py`
- `engine.py`
- 复用已有实现：`util/utils.py` 中 `ModelEma`

### 关键机制
- 训练时每个 iteration 完成优化器更新后执行 `ema_m.update(model)`。
- 评估时可切换使用 EMA 权重或当前在线权重。
- checkpoint 保存 `ema_model`，支持断点续训恢复。

### 启停参数（`main.py`）
- `--enable_ema`：开启 EMA
- `--ema_decay`：EMA 衰减系数（默认 `0.9997`）
- `--ema_device`：EMA 存储设备（可设 `cpu` 降低显存占用）
- `--use_ema_for_eval`：验证/测试时使用 EMA 模型

### 参考论文
- Mean Teacher（EMA 教师思想）：https://arxiv.org/abs/1703.01780
- Stochastic Weight Averaging（权重平均思想）：https://arxiv.org/abs/1803.05407

---

## 2.2 创新点 B：小目标感知回归损失加权（面向 HRSID 小目标）
### 动机
HRSID 中大量舰船目标尺度偏小，小目标定位误差对检测结果更敏感。  
标准 L1/GIoU 回归损失对不同尺度目标权重一致，不利于强化小目标定位。

### 实现位置
- `models/dab_deformable_detr/dab_deformable_detr.py` 的 `SetCriterion`
- `models/DAB_DETR/DABDETR.py` 的 `SetCriterion`

### 关键机制
- 在 `loss_boxes` 中基于目标框归一化面积 `area = w*h` 计算权重：
  - `weight = 1 + alpha * (1 - area)^power`
- 对 `loss_bbox` 和 `loss_giou` 做逐框加权求和。
- 默认关闭，不影响原始损失定义。

### 启停参数（`main.py`）
- `--enable_small_object_reweight`：开启小目标加权
- `--small_object_reweight_alpha`：加权强度（默认 `1.0`）
- `--small_object_reweight_power`：非线性指数（默认 `0.5`）

### 参考论文
- Focal Loss（困难样本重加权思想）：https://arxiv.org/abs/1708.02002
- Libra R-CNN（回归损失重加权/平衡思想）：https://arxiv.org/abs/1904.02701

---

## 2.3 创新点 C：骨干网络渐进解冻（迁移学习友好）
### 动机
微调阶段直接全量更新 backbone 可能导致早期灾难性遗忘。  
先冻结 backbone、后解冻，有助于检测头先适配 SAR 分布，再联合优化特征提取层。

### 实现位置
- `main.py`

### 关键机制
- epoch 级控制 backbone 参数 `requires_grad`。
- 保留原始参数可训练状态，避免错误覆盖本来就冻结的参数。

### 启停参数（`main.py`）
- `--enable_backbone_warmup`：开启骨干渐进解冻
- `--backbone_warmup_epochs`：冻结 epoch 数（例如 `5`）

### 参考论文
- ULMFiT（Discriminative fine-tuning + gradual unfreezing）：https://arxiv.org/abs/1801.06146

---

## 2.4 创新点 D：多尺度 TTA 评估融合（推理增强）
### 动机
SAR 舰船尺度变化大，多尺度推理可提高召回。  
通过多尺度预测 + NMS 融合，可在不改训练流程情况下提升推理表现。

### 实现位置
- `engine.py`

### 关键机制
- 验证/测试时对输入做多尺度前向推理。
- 收集各尺度预测结果并拼接。
- 使用 `batched_nms` 做类别感知融合，最后截断 `topk`。

### 启停参数（`main.py`）
- `--enable_tta_eval`：开启 TTA 评估
- `--tta_scales`：尺度列表（默认 `1.0 1.15 1.3`）
- `--tta_score_thresh`：融合前置信度阈值
- `--tta_nms_iou_thresh`：融合 NMS 阈值
- `--tta_topk`：每图最终保留框数

### 参考论文
- SNIP（多尺度训练/测试思想）：https://arxiv.org/abs/1711.08189
- Soft-NMS（预测框后处理融合思想）：https://arxiv.org/abs/1704.04503

---

## 2.5 创新点 F：SAR 舰船形状先验损失（长宽比约束）
### 动机
舰船在 SAR 图像中常呈现细长结构。仅使用 L1 + GIoU 时，模型对“长宽比结构”学习不够显式。  
新增 `loss_shape`，对匹配框的 `log(w/h)` 进行约束，强化舰船几何形状学习。

### 实现位置
- `models/dab_deformable_detr/dab_deformable_detr.py`
- `models/DAB_DETR/DABDETR.py`
- `main.py`

### 关键机制
- 在 `loss_boxes` 中新增：
  - `src_ratio = log(w_pred / h_pred)`
  - `tgt_ratio = log(w_gt / h_gt)`
  - `loss_shape = L1(src_ratio, tgt_ratio)`
- 支持与小目标加权联动，默认关闭。

### 启停参数（`main.py`）
- `--enable_sar_shape_prior_loss`
- `--sar_shape_prior_loss_coef`

### 参考论文
- 远感检测中长宽比分布建模思想（SARA）：https://doi.org/10.3390/rs13071318  
- Aspect-ratio sensitive oriented detection（ARS-DETR）：https://arxiv.org/abs/2303.04989

---

## 2.6 创新点 G：频域显著信号保留引导的 Query 初始化（DAB-Deformable 专用）
### 动机
在 DAB-Deformable-DETR 中，默认 `refpoint_embed` 为全局可学习参数，与当前图像内容弱耦合。  
针对 SAR 中“亮斑目标 + 稀疏背景”特性，引入频域显著性引导，让初始 query 坐标更靠近高响应区域，以加快收敛并改善定位精度。

### 参考与借鉴来源
- 论文：TransDeno（你指定）：https://arxiv.org/abs/2406.02833
- DenoDet/GrokSAR 公开实现（借鉴“显著信号保留”的频域处理逻辑）：
  - `GrokSAR/groksar/models/backbones/FFTresnet.py` 中 `FFTTransformerattentionlayer`
  - 核心流程：`FFT -> 幅相分解 -> 相位对齐 -> IFFT -> ReLU`

### 实现位置
- `models/dab_deformable_detr/dab_deformable_detr.py`
- `models/dab_deformable_detr/deformable_transformer.py`
- `main.py`

### 关键机制
1. 在 `use_dab=True 且 two_stage=False` 时，新增可选“图像自适应 query 初始化”。  
2. 从选定特征层（默认 level-0）构建频域显著图：  
   - `fft2 + fftshift`  
   - 幅相分解（`amplitude / phase`）  
   - 相位正余弦归一化对齐（Phase Alignment）  
   - 复频谱重建后 `ifft2` 回空间域，`ReLU`  
   - 用 `mean + max` 形成显著性响应（与 DenoDet 注意力构造风格一致）  
3. 可选高通支路（`saliency_highpass_radius_ratio`）强化亮斑和边缘响应。  
4. 在有效区域（mask 非 padding）上取 Top-K 响应点，将其映射为 query 初始 `(x, y)`。  
5. `w,h` 仍沿用原 `refpoint_embed` 的可学习先验，保证与 DAB 迭代框细化兼容。  
6. 为兼容该改动，transformer 支持 batch 级 query embedding（`[B, Nq, D+4]`）。

### 启停参数（`main.py`）
- `--enable_saliency_query_init`：开启频域显著引导 query 初始化
- `--saliency_query_feature_level`：用于生成显著图的特征层索引（默认 `0`）
- `--saliency_highpass_radius_ratio`：高通半径比例（默认 `0.15`，设为 `0` 可关闭）

### 使用建议
- 推荐先在 `exp03_loss_sar` 或 `exp04_full` 上追加一组：
  - `--enable_saliency_query_init`
- 若训练早期不稳定，可先设置：
  - `--saliency_highpass_radius_ratio 0`
  - 再逐步调到 `0.1~0.2`

---

## 3. 额外工程性修复
- 在 `engine.py` 的非 AMP 路径补充了 `optimizer.zero_grad()`，避免梯度无意累积导致训练不稳定。

---

## 4. 消融实验建议
考虑到你 GPU 和预算有限，主实验改为仅 5 组（含基线），并将创新点按层级整合，减少实验次数。

分组原则：
- Query 初始化单独成组（G）
- 模型训练层创新合并成单组（A+C）
- 损失层创新合并成单组（B+F）

最终只保留：
1. `exp00_baseline`：基线  
2. `exp01_query_init`：Query 初始化（G）  
3. `exp02_model_stab`：模型训练层（A+C）  
4. `exp03_loss_sar`：损失层（B+F）  
5. `exp04_full`：全量轻量组合（G+A+C+B+F）  

`D(TTA)` 不计入这 5 组，仅在最终最优组上额外评估一次。

已从主消融中删除/合并（为节省算力）：
- 删除 A、B、C、F、G 的逐个单独消融，不再分别占用一次完整训练。
- 删除多种交叉组合（如 AB、AC、ABC、SAR-only 分裂组），统一折叠进 4 个非基线主组。
- `TTA` 从训练消融中剥离，仅做最终一次评估补充。

---

## 5. 标准化消融脚本（可直接复现实验表格）
已新增脚本：`tools/run_ablation_table.sh`。

### 5.1 设计说明
- 脚本完整复用 `train.sh` 的基础参数，不修改任何原始超参数。
- 仅通过附加开关启用整合后的模块组合，满足低预算消融要求。
- 输出目录统一到：`output/dab_deformable_detr/ablation_table_2026_lite/`

### 5.2 快速使用
```bash
# 查看实验计划
bash tools/run_ablation_table.sh plan

# 一键跑全套 5 组训练消融（exp00 ~ exp04）
bash tools/run_ablation_table.sh all

# 单独运行某个实验
bash tools/run_ablation_table.sh exp03_loss_sar

# 对某个训练结果做 TTA 评估
bash tools/run_ablation_table.sh eval_tta exp04_full
```

若环境中没有 `python` 命令，可用：
```bash
PYTHON_BIN=python3 bash tools/run_ablation_table.sh all
```

### 5.3 标准实验组定义
| 实验ID | 分类 | 实验名 | 模块组合 | 新增参数 |
|---|---|---|---|---|
| exp00_baseline | 基线组 | baseline | 无 | 无 |
| exp01_query_init | Query初始化组 | query_init | G | `--enable_saliency_query_init` |
| exp02_model_stab | 模型训练层组 | model_stab | A + C | `--enable_ema --use_ema_for_eval --enable_backbone_warmup` |
| exp03_loss_sar | 损失层组（SAR相关） | loss_sar | B + F | `--enable_small_object_reweight --enable_sar_shape_prior_loss` |
| exp04_full | 综合组 | full | G + A + C + B + F | 全部整合参数 |

建议在 `exp04_full` 基础上额外执行 `eval_tta`，作为“仅推理阶段增强”的补充结果。

### 5.4 论文表格模板（建议）
| Exp | A(EMA) | B(Small) | C(Warmup) | G(QueryInit) | F(SAR Shape) | D(TTA, eval) | mAP@[.5:.95] | AP50 | AP75 | 参数量(M) | 训练时长(h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp00 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |  |  |  |  |  |
| exp01_query_init | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |  |  |  |  |  |
| exp02_model_stab | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |  |  |  |  |  |
| exp03_loss_sar | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |  |  |  |  |  |
| exp04_full | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |  |  |  |  |  |
| exp04_full+TTA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |

---

## 6. 训练建议与参数设置建议（不改 train.sh 基线）
以下建议用于你后续扩展实验；当前 `tools/run_ablation_table.sh` 未改动 `train.sh` 原参数。

### 6.1 基线参数（保持不变）
- `batch_size=4`
- `epochs=50`
- `lr=1e-4`
- `lr_backbone=1e-5`
- `lr_T_max=100`
- `lr_eta_min=1e-6`
- `focal_alpha=0.2`
- `bbox_loss_coef=6`
- `giou_loss_coef=3`

### 6.2 推荐的调参顺序
1. 先跑 5 组固定实验：`exp00` 到 `exp04`。  
2. 从 `exp04_full` 出发做二阶段微调。  
3. 先调 Query 初始化相关：`saliency_query_feature_level`、`saliency_highpass_radius_ratio`。  
4. 再调稳定性相关：`ema_decay`、`backbone_warmup_epochs`。  
5. 最后只对最优权重做一次 `TTA` 评估。

### 6.3 参数建议范围（旁注建议，不自动修改）
- A(EMA)
  - 当前：`ema_decay=0.9997`
  - 可尝试：`0.9995 / 0.9997 / 0.9999`
  - 旁注：batch 较小时 `0.9997` 往往更稳。
- B(Small-object reweight)
  - 当前：`alpha=1.0, power=0.5`
  - 可尝试：`alpha ∈ [0.5, 1.5]`，`power ∈ [0.3, 0.8]`
  - 旁注：若 AP50 上升而 AP75 下降，通常是加权过强。
- C(Backbone warmup)
  - 当前：`warmup_epochs=5`
  - 可尝试：`3 / 5 / 8`
  - 旁注：数据集较小时，`5` 通常是更稳妥的起点。
- D(TTA eval)
  - 当前：`scales=1.0,1.15,1.3`
  - 可尝试：`1.0,1.1,1.2`（更快）或 `1.0,1.2,1.4`（更强但更慢）
  - 旁注：`tta_nms_iou_thresh` 建议在 `0.55~0.65` 小范围调整。
- G(QueryInit)
  - 当前：`feature_level=0`，`highpass_radius_ratio=0.15`
  - 可尝试：`feature_level ∈ {0,1}`，`radius_ratio ∈ {0, 0.1, 0.15, 0.2}`
  - 旁注：若训练早期振荡，可先将 `radius_ratio` 设为 `0`。
- F(SAR Shape)
  - 当前：`sar_shape_prior_loss_coef=0.3`
  - 可尝试：`0.1 / 0.3 / 0.5`
  - 旁注：若 AP75 下降，先将系数降到 `0.1`。

### 6.4 训练过程控制建议
- 固定随机种子：建议至少 `seed=42/123/3407` 三次重复，报告均值与标准差。
- 断点续训：优先用 `checkpoint.pth`，本项目已支持 EMA 权重同步恢复。
- 日志记录：保留每次实验 `config.json + log.txt + eval/latest.pth`，便于复盘。
- 公平对比：消融时只允许单变量变化，保持数据路径与增强流程一致。

### 6.5 资源与时长建议
- 单卡显存不足时：优先保持 `batch_size=4`，必要时再降到 `2`。
- 若降到 `batch_size=2`（旁注建议）：建议把学习率按线性规则同步降到 `5e-5`（不自动修改）。
- 若训练时间允许（旁注建议）：可增加到 `epochs=75~100` 做上限探索，但不纳入与 50 epoch 的主消融同表比较。

---

## 7. 可拓展方向（下一步）
- 引入旋转框检测头（适配 SAR 舰船长条方向性）。
- 将小目标权重扩展为动态课程策略（随 epoch 渐变）。
- 引入伪标签半监督微调（充分利用未标注 SAR 图像）。

---

## 8. 基线与本项目直接相关论文（建议在毕设中引用）
- DAB-DETR（你的方法主线）：https://arxiv.org/abs/2201.12329  
- Deformable DETR（你的主干变体基础）：https://arxiv.org/abs/2010.04159  
- DETR（端到端检测基础）：https://arxiv.org/abs/2005.12872  
