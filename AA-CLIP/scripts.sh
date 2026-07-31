# =====================================================================
# AA-CLIP 复现脚本：训练 + 测试 MVTec-AD 与私有数据集
# =====================================================================

# ---------- 1. 训练 ----------
# MVTec-AD (15 个类别，full-shot 训练)
python train.py --dataset MVTec --training_mode full_shot --save_path ./ckpt/mvtec

# 私有数据集 (自动包含 constants.py 中 CLASS_NAMES["Private"] 的所有 object)
python train.py --dataset Private --training_mode full_shot --save_path ./ckpt/private

# ---------- 2. 测试 ----------
# MVTec-AD
python test.py --dataset MVTec --save_path ./ckpt/mvtec

# 私有数据集
python test.py --dataset Private --save_path ./ckpt/private
