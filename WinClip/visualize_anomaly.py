"""
WinCLIP 异常检测热力图可视化脚本

用法:
    # few-shot (正常图片目录中有图片作为正样本)
    python visualize_anomaly.py --anomaly-image ./test.png --normal-dir ./normal_images --category bottle

    # zero-shot (正常图片目录为空或不存在图片)
    python visualize_anomaly.py --anomaly-image ./test.png --normal-dir ./empty_dir --category bottle

输入:
    1. --anomaly-image : 一张待检测的异常图片
    2. --normal-dir    : 一个目录, 里面放正常(无缺陷)图片作为正样本参考
                         - 若目录中有图片 -> few-shot 模式 (文本+视觉双重判别)
                         - 若目录为空/不存在图片 -> zero-shot 模式 (仅用文本判别)

输出:
    - 原图 + 异常热力图 + 叠加图 的三联对比图 (png)
    - 单独的热力图与叠加图
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib
# 不强制使用 Agg 后端, 以便 plt.show() 能弹出窗口; 若环境不支持可注释下一行
matplotlib.use('TkAgg', force=False)
import matplotlib.pyplot as plt

from WinCLIP import WinClipAD
## 007 021

def parse_args():
    parser = argparse.ArgumentParser(description='WinCLIP 异常检测热力图可视化')
    parser.add_argument('--anomaly-image', type=str, default='./test/test3/abnorm/001.png',
                        help='待检测的异常图片路径')
    parser.add_argument('--normal-dir', type=str, default='./test/test3/normal',
                        help='正常参考图片所在目录 (空目录=zero-shot)')
    parser.add_argument('--category', type=str, default='object',
                        help='物体类别名称, 用于文本提示 (如 bottle/cable/wood)。默认 object')
    parser.add_argument('--output', type=str, default='./heatmap_result.png',
                        help='可视化结果保存路径 (默认 ./heatmap_result.png)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cuda', 'cpu'],
                        help='运行设备 (auto 自动选择)')
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained-dataset', type=str, default='laion400m_e32')
    parser.add_argument('--scales', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400,
                        help='输出热力图分辨率 (边长)')
    parser.add_argument('--no-show', action='store_true',
                        help='不弹窗显示, 仅保存文件')
    return parser.parse_args()


def collect_normal_images(normal_dir):
    """从目录收集正常图片路径, 返回排序后的列表; 无图片返回 None"""
    if not os.path.isdir(normal_dir):
        print(f'[INFO] 正常图片目录不存在: {normal_dir} -> zero-shot 模式')
        return None
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif', '*.webp')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(normal_dir, ext)))
        # 大小写兼容
        paths.extend(glob.glob(os.path.join(normal_dir, ext.upper())))
    # 去重 + 排序
    paths = sorted(set(paths))
    if len(paths) == 0:
        print(f'[INFO] 目录 {normal_dir} 中未找到图片 -> zero-shot 模式')
        return None
    return paths


def main():
    args = parse_args()

    # ---- 设备 ----
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f'[INFO] 使用设备: {device}')

    # ---- 校验异常图片 ----
    if not os.path.isfile(args.anomaly_image):
        print(f'[ERROR] 找不到异常图片: {args.anomaly_image}')
        sys.exit(1)

    # ---- 构建模型 ----
    # WinClipAD 内部会把 precision 硬编码为 'fp16', 这里先按其默认构建
    kwargs = dict(
        out_size_h=args.resolution,
        out_size_w=args.resolution,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        scales=args.scales,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
    )
    print('[INFO] 正在构建 WinCLIP 模型 (首次运行会自动下载约 700MB 预训练权重到 ~/.cache/clip)...')
    model = WinClipAD(**kwargs)
    model = model.to(device)

    # ---- CPU 兼容: 将 fp16 模型转回 fp32 (fp16 在 CPU 上慢且部分算子不支持) ----
    if device == 'cpu':
        print('[INFO] 检测到 CPU 运行, 将模型转为 fp32 精度以提升兼容性...')
        model.precision = 'fp32'
        model.model.float()

    # ---- 文本特征库 (zero-shot 与 few-shot 都需要) ----
    print(f'[INFO] 构建文本特征库, 类别: {args.category}')
    model.build_text_feature_gallery(args.category)

    # ---- 视觉特征库 (仅当有正样本图片时) ----
    normal_paths = collect_normal_images(args.normal_dir)
    model.eval_mode()
    if normal_paths is not None:
        print(f'[INFO] 加载 {len(normal_paths)} 张正常参考图片 -> few-shot 模式')
        normal_tensors = []
        for p in normal_paths:
            img = Image.open(p).convert('RGB')
            normal_tensors.append(model.transform(img))
        with torch.no_grad():
            # 一次性送入 (正样本通常不多; 数量很多时可分批, 此处从简)
            batch = torch.stack(normal_tensors, dim=0).to(device)
            model.build_image_feature_gallery(batch)
        print('[INFO] 视觉特征库构建完成')
    else:
        print('[INFO] 跳过视觉特征库 (zero-shot)')

    # ---- 处理异常图片 ----
    print(f'[INFO] 处理异常图片: {args.anomaly_image}')
    pil_img = Image.open(args.anomaly_image).convert('RGB')
    input_tensor = model.transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        score_list = model(input_tensor)

    anomaly_map = score_list[0]  # numpy (H, W), 值越大越异常

    # ---- 可视化 ----
    res = args.resolution
    # 原图缩放到与热力图相同尺寸, 便于对齐叠加
    orig_np = np.array(pil_img.resize((res, res), Image.BILINEAR))

    # 归一化热力图到 0-255
    am = anomaly_map.astype(np.float32)
    am_min, am_max = am.min(), am.max()
    am_norm = (am - am_min) / (am_max - am_min + 1e-8) * 255.0
    am_uint8 = am_norm.astype(np.uint8)

    # 伪彩色热力图 (cv2.applyColorMap 输出 BGR)
    heatmap_bgr = cv2.applyColorMap(am_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 叠加图: 热力图 0.5 + 原图 0.5
    overlay_rgb = cv2.addWeighted(heatmap_rgb, 0.5, orig_np, 0.5, 0)

    # ---- 三联图绘制 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')

    im = axes[1].imshow(heatmap_rgb)
    axes[1].set_title('Anomaly Heatmap', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay_rgb)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    # 保存主图
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'[INFO] 三联可视化图已保存: {args.output}')

    # 额外保存单张热力图与叠加图 (BGR, 供 cv2 直接查看)
    base, _ = os.path.splitext(args.output)
    heatmap_path = base + '_heatmap.png'
    overlay_path = base + '_overlay.png'
    cv2.imwrite(heatmap_path, heatmap_bgr)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    print(f'[INFO] 热力图已保存: {heatmap_path}')
    print(f'[INFO] 叠加图已保存: {overlay_path}')

    if not args.no_show:
        print('[INFO] 弹出显示窗口 (关闭窗口后程序结束)...')
        plt.show()

    print('[INFO] 完成!')


if __name__ == '__main__':
    main()
