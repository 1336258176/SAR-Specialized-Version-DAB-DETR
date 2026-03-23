import io
import sys
import torch
import torch.nn as nn
from torchinfo import summary
from models.DAB_DETR.backbone import Backbone, Joiner, FrozenBatchNorm2d
from models.DAB_DETR.position_encoding import PositionEmbeddingSine
from models.DAB_DETR.transformer import Transformer
from models.DAB_DETR.DABDETR import DABDETR
from util.misc import NestedTensor


class DABDETRWrapper(nn.Module):
    """将普通 Tensor 转为 NestedTensor，方便 torchinfo 进行前向追踪"""

    def __init__(self, model: DABDETR):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]，构造全有效像素的 mask（无 padding）
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)
        return self.model(NestedTensor(x, mask))


def count_parameters(model):
    """统计各模块的参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_param_table(model):
    """打印各主要子模块的参数量表格"""
    print("\n" + "=" * 60)
    print(f"{'模块':<30} {'参数量':>15} {'可训练参数量':>15}")
    print("=" * 60)
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<28} {total:>15,} {trainable:>15,}")
    print("-" * 60)
    total, trainable = count_parameters(model)
    print(f"  {'全部参数合计':<28} {total:>15,} {trainable:>15,}")
    print("=" * 60)


if __name__ == "__main__":
    # ── 构建模型 ──────────────────────────────────────────────
    # 骨干网络需要用 Joiner 包装以同时输出特征和位置编码
    _backbone = Backbone("resnet50", True, False, False, FrozenBatchNorm2d)
    _pos_enc = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    backbone = Joiner(_backbone, _pos_enc)
    backbone.num_channels = _backbone.num_channels
    transformer = Transformer(
        d_model=256,
        dropout=0.0,
        nhead=8,
        num_queries=300,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=4,
        activation="prelu",
        num_patterns=0,
    )
    model = DABDETR(
        backbone,
        transformer,
        num_classes=2,
        num_queries=300,
        num_dec_layers=6,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=False,
    )
    model.eval()

    # ── 3. 用包装器让 torchinfo 接受标准 Tensor 输入 ─────────
    wrapper = DABDETRWrapper(model)
    wrapper.eval()

    output_path = "model_summary.txt"
    orig_stdout = sys.stdout

    class Tee(io.TextIOBase):
        """同时写入控制台和文件的流"""

        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
            return len(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(output_path, "w", encoding="utf-8") as f:
        tee = Tee(orig_stdout, f)
        sys.stdout = tee

        # ── 1. PyTorch 原生网络结构 ───────────────────────────
        print("\n" + "=" * 60)
        print("           DAB-DETR 网络结构（PyTorch 原生）")
        print("=" * 60)
        print(model)

        # ── 2. 各子模块参数量统计 ─────────────────────────────
        print_param_table(model)

        # ── 3. torchinfo 详细摘要 ─────────────────────────────
        print("\n" + "=" * 60)
        print("      DAB-DETR torchinfo 详细摘要（input: 1×3×640×640）")
        print("=" * 60)
        summary(
            wrapper,
            input_size=(1, 3, 640, 640),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            depth=4,
            verbose=1,
        )

        sys.stdout = orig_stdout

    print(f"\n[已保存至 {output_path}]")
