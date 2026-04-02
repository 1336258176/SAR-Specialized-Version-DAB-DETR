# DAB-Deformable-DETR 架构详解（基于本仓库实现）

本文档对应实现：
- `models/dab_deformable_detr/dab_deformable_detr.py`
- `models/dab_deformable_detr/deformable_transformer.py`
- `models/dab_deformable_detr/backbone.py`
- `models/dab_deformable_detr/matcher.py`
- `models/dab_deformable_detr/position_encoding.py`
- 训练参数默认值来自 `main.py`

## 1. 总体结构

DAB-Deformable-DETR 可以看作三段：
1. `Backbone`（ResNet）提取多尺度特征。
2. `Neck`（轻量投影层）把各尺度通道统一到 `hidden_dim=256`，并补齐特征层级。
3. `Transformer`（Deformable Encoder + DAB Decoder）做全局建模和集合预测，最后输出分类与框。

主干前向（简化）如下：
- 输入：`samples`（`NestedTensor`，含图像和 padding mask）
- Backbone 输出：多尺度 `features` + 对应位置编码 `pos`
- Neck 输出：`srcs/masks/pos`（统一到 256 通道）
- Transformer 输出：
  - `hs`：各解码层 query 特征
  - `init_reference/inter_references`：初始与逐层更新的 reference points
- Head：每层 decoder 都有 `class_embed` 和 `bbox_embed`（iterative refinement），最终输出：
  - `pred_logits` `[B, Nq, C]`
  - `pred_boxes` `[B, Nq, 4]`（`cx,cy,w,h`，归一化到 `[0,1]`）

默认关键超参数（`main.py`）：
- `backbone=resnet50`
- `hidden_dim=256`
- `enc_layers=6`, `dec_layers=6`
- `nheads=8`
- `num_queries=300`
- `num_feature_levels=4`
- `enc_n_points=4`, `dec_n_points=4`
- `dropout=0.0`
- `dim_feedforward=2048`

## 2. Backbone（特征提取）

实现位置：`models/dab_deformable_detr/backbone.py`

### 2.1 主体网络
- 使用 torchvision ResNet（默认 `resnet50`）。
- BN 使用 `FrozenBatchNorm2d`（参数与统计量冻结，提升稳定性）。
- 当 `num_feature_levels > 1` 时，返回中间层：`layer2/layer3/layer4`。
  - 对应 stride：`[8,16,32]`
  - 对应通道：`[512,1024,2048]`

### 2.2 可训练策略
- 若 `lr_backbone <= 0`，backbone 全冻结。
- 否则默认训练 `layer2/layer3/layer4`，更浅层冻结。

### 2.3 位置编码
- `Joiner(backbone, position_embedding)` 同时返回特征和位置编码。
- 默认是 `PositionEmbeddingSine(normalize=True)`。
- 每个尺度都有对应位置编码，并转到对应 dtype。

## 3. Neck（多尺度对齐与扩展）

这份实现没有传统 FPN；Neck 是 `input_proj`：
- 对 backbone 原生输出层（3 层）使用 `1x1 Conv + GroupNorm(32)`，通道统一到 256。
- 若 `num_feature_levels` 大于 backbone 输出层数（默认 4 > 3），额外添加 `3x3 Conv(stride=2,pad=1) + GN` 逐级下采样，补出更低分辨率层。

因此默认 4 个尺度来自：
1. C3（stride 8）投影
2. C4（stride 16）投影
3. C5（stride 32）投影
4. 在上一层基础上再下采样得到的额外层（约 stride 64）

每个尺度都维护 mask，并插值到相同空间大小，保证 Deformable Attention 可感知有效区域。

## 4. Encoder（Deformable Transformer Encoder）

实现位置：`deformable_transformer.py` 中 `DeformableTransformerEncoder`

### 4.1 输入组织
- 每个尺度特征 `[B,C,H,W]` 展平为 `[B,HW,C]`，再跨尺度拼接成 `[B, sum(H_lW_l), C]`。
- 记录：
  - `spatial_shapes = [[H1,W1],...,[HL,WL]]`
  - `level_start_index`（每层在拼接序列中的起始偏移）
  - `valid_ratios`（每层有效宽高比例，用于处理 padding）

### 4.2 参考点
Encoder 为所有 token 构建归一化 reference points：
- 每层网格中心 `(x+0.5, y+0.5)` 按 valid ratio 归一化
- 形状约为 `[B, Len, L, 2]`

### 4.3 每层结构
每个 Encoder layer：
1. `MSDeformAttn` 自注意力（多尺度、稀疏采样）
2. 残差 + LayerNorm
3. FFN（`Linear(256->2048)->act->Linear(2048->256)`）
4. 残差 + LayerNorm

默认 6 层堆叠。

## 5. Decoder（DAB + Deformable Decoder）

实现位置：`DeformableTransformerDecoder`

### 5.1 Query 初始化（DAB 核心）
本仓库构建 DAB-Deformable 时固定 `use_dab=True`。

非 two-stage 情况下：
- `tgt_embed`: `Embedding(num_queries, 256)`，作为 query content。
- `refpoint_embed`: `Embedding(num_queries, 4)`，作为动态 anchor（`cx,cy,w,h` 的未归一化参数）。
- 拼接为 `query_embeds=[tgt_embed, refanchor]` 送入 transformer。

可选：
- `num_patterns > 0` 时会复制 query 并叠加 pattern embedding。
- `random_refpoints_xy` 可随机初始化 `(x,y)` 并冻结。

### 5.2 DAB 的 query positional 生成
在每个 decoder 层中：
1. 从当前 `reference_points` 生成正弦嵌入（默认使用 `(x,y)`）。
2. 通过 `ref_point_head` MLP 得到 `raw_query_pos`。
3. 第 0 层直接用；后续层用 `query_scale(output)` 自适应缩放。

直观上：query 的位置先验不是固定向量，而是由当前 reference 动态产生。

### 5.3 每层结构
每个 Decoder layer：
1. Self-attention（`nn.MultiheadAttention`）
2. Cross-attention（`MSDeformAttn`，围绕 reference points 跨尺度采样）
3. FFN

### 5.4 逐层框精修（iterative refinement）
若开启 `with_box_refine`（本实现默认 True）：
- 每层有独立的 `bbox_embed[lid]` 和 `class_embed[lid]`。
- `bbox_embed` 预测的是增量，与 `inverse_sigmoid(reference)` 相加后再 `sigmoid` 得到新框。
- 新 reference points `detach` 后传给下一层。

这使 decoder 形成“逐层细化框”的流程。

### 5.5 Two-stage 分支（可选）
若 `--two_stage`：
- Encoder 输出上先生成 proposal，选 top-k（`num_queries`）作为 decoder 初始 reference。
- 同时会额外产生 `enc_outputs` 并参与损失。

默认训练一般为 one-stage（`two_stage=False`）。

## 6. Detection Heads 与输出

在 `DABDeformableDETR` 中：
- 分类头：`Linear(256, num_classes)`（logits，经 sigmoid 用于多标签 focal）
- 回归头：`MLP(256,256,4,3)`

输出：
- 主输出：最后一层 decoder 的 `pred_logits/pred_boxes`
- 辅助输出：`aux_outputs`（前 `dec_layers-1` 层）
- two-stage 时还有 `enc_outputs`

推理后处理 `PostProcess`：
- 对 `pred_logits.sigmoid()` 在 `Nq*C` 上取 top-100
- 根据索引恢复 label 与 box
- box 从归一化坐标缩放回原图尺寸

## 7. 匹配机制（Hungarian Matcher）

实现位置：`matcher.py`

对每张图，构建代价矩阵并做一对一匹配：
- 分类代价：基于 focal 形式的正负项差值
- 框 L1 代价：`cdist(pred_box, tgt_box, p=1)`
- GIoU 代价：`-GIoU(pred_box, tgt_box)`

总匹配代价：
`C = w_cls * cost_class + w_l1 * cost_bbox + w_giou * cost_giou`

默认匹配权重：
- `set_cost_class=2`
- `set_cost_bbox=5`
- `set_cost_giou=2`

## 8. 损失函数（SetCriterion）

实现位置：`dab_deformable_detr.py` 的 `SetCriterion`

训练先做 Hungarian 匹配，再只对 matched 对监督（未匹配 query 作为背景）。

### 8.1 分类损失 `loss_ce`
- 使用 `sigmoid_focal_loss`，`alpha=args.focal_alpha(默认0.25)`，`gamma=2.5`。
- one-hot 去掉 no-object 通道后计算。
- 实现里乘了 `num_queries` 作为缩放。

### 8.2 框回归损失 `loss_bbox`
- 对 matched 预测框与 GT 框做 L1：
`L_bbox = |b - b*|`
- 默认权重系数：`bbox_loss_coef=5`

### 8.3 几何损失 `loss_giou`
- `L_giou = 1 - GIoU(pred, gt)`
- 默认权重系数：`giou_loss_coef=2`

### 8.4 小目标重加权（本仓库增强项，可选）
若 `--enable_small_object_reweight`：
- 对每个 matched GT 计算面积 `a = w*h`（归一化面积）
- 样本权重：
`rw = 1 + alpha * (1-a)^power`
  - `alpha=small_object_reweight_alpha`（默认 1.0）
  - `power=small_object_reweight_power`（默认 0.5）
- 该权重会乘到 `loss_bbox` 与 `loss_giou`（以及可选 shape loss）上。

含义：面积越小，权重通常越大，强化小目标学习。

### 8.5 SAR 形状先验损失（本仓库增强项，可选）
若 `--enable_sar_shape_prior_loss`：
- 定义长宽比对数：`r = log(w/h)`
- `loss_shape = L1(r_pred, r_gt)`
- 也使用上面的 box reweight
- 权重系数：`sar_shape_prior_loss_coef`（默认 0.3）

### 8.6 辅助损失与编码器损失
- 若 `aux_loss=True`（默认开启），每个中间 decoder 层都复制一套损失，key 带后缀 `_i`。
- two-stage 时还会对 `enc_outputs` 计算同类损失，key 后缀 `_enc`。

### 8.7 总损失
总损失是各项加权和：
- 主层：`loss_ce + 5*loss_bbox + 2*loss_giou (+ 0.3*loss_shape 可选)`
- 加上各 decoder 中间层和可选 encoder 层对应项（系数相同，名字带后缀）。

## 9. 你这个项目中的“Backbone/Neck/Encoder/Decoder/Loss”一句话总结

- Backbone：ResNet50（FrozenBN），默认输出 C3/C4/C5 多尺度。
- Neck：`1x1 Conv+GN` 对齐通道 + `3x3 s2 Conv+GN` 扩展到 4 个尺度，无传统 FPN 融合。
- Encoder：6 层多尺度可变形自注意力（MSDeformAttn）+ FFN。
- Decoder：6 层 DAB query + deformable cross-attn + iterative box refinement。
- Loss：Hungarian 一对一匹配后，用 focal cls + L1 box + GIoU，外加你仓库的“小目标重加权”和“shape prior”可选项。

