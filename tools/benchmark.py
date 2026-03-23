import torch
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser
from models.DAB_DETR import build_DABDETR
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.utils.benchmark import Timer


def get_args():
    parser = get_args_parser()
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 初始化模型中...")
    assert args.modelname == "dab_detr"
    model, _, _ = build_DABDETR(args)

    model.to(device)
    model.eval()

    # 针对 SAR 图像，通常切片大小为 800x800
    print("📦 构造测试张量 (Batch Size=1, 3x800x800)...")
    dummy_input = torch.randn(1, 3, 800, 800).to(device)

    # ==========================================
    # 1. 复杂度 Benchmark (Params & FLOPs)
    # ==========================================
    print("\n" + "=" * 40)
    print("📊 1. 复杂度与参数量评估 (FVCore)")
    print("=" * 40)

    flops = FlopCountAnalysis(model, dummy_input)
    # 忽略一些不受支持的极其微小的算子警告
    flops.unsupported_ops_warnings(False)

    macs = flops.total()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 打印详细的参数表（可以清晰看到Backbone和Transformer各占多少）
    # print(parameter_count_table(model))

    print(f"✅ 计算量 (GFLOPs): {macs / 1e9:.2f} G")
    print(f"✅ 参数量 (Params): {params / 1e6:.2f} M")

    # # ==========================================
    # # 2. 速度 Benchmark (FPS & Latency)
    # # ==========================================
    # print("\n" + "=" * 40)
    # print("🏎️ 2. 推理速度评估 (PyTorch Benchmark)")
    # print("=" * 40)

    # # PyTorch 官方的高精度测速工具
    # timer = Timer(
    #     stmt="model(x)",
    #     globals={"model": model, "x": dummy_input},
    #     num_threads=1,
    # )

    # # 自动进行 Warm-up 并执行多次测速以获得稳定均值
    # print("正在进行热身和多轮测试，请耐心等待...")
    # measurement = timer.blocked_autorange(min_run_time=3.0)

    # # 解析结果
    # latency_ms = measurement.mean * 1000
    # fps = 1.0 / measurement.mean

    # print(measurement)
    # print(f"✅ 平均延迟 (Latency): {latency_ms:.2f} ms")
    # print(f"✅ 吞吐量 (FPS):      {fps:.2f} img/s")


if __name__ == "__main__":
    main()
