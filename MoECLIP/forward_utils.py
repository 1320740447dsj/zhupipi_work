import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim
from scipy.ndimage import gaussian_filter, label as cc_label, find_objects

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


# ================================================================================================
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc


def calculate_aupro(pixel_label: np.ndarray, pixel_preds: np.ndarray, max_fpr: float = 0.3):
    """
    计算 AUPRO (Per-Region Overlap) 指标。
    思路:对每张图的 ground truth mask 做连通域提取,把每个连通域视为一个独立"正样本区域",
    以全图其余像素为"负样本区域"计算 ROC 曲线,取 FPR<=max_fpr 区间的 AUC 并对所有区域求平均。

    参数:
        pixel_label: shape (N, H, W) 或 (N, 1, H, W), 二值 ground truth (0/1)
        pixel_preds: shape (N, H, W) 或 (N, 1, H, W), 异常预测分数 (已归一化到 [0,1])
        max_fpr:     AUPRO 截断的假阳性率上限,工业界惯例为 0.3
    返回:
        aupro: 标量,所有区域的平均 AUPRO (0~1)
    """
    # 自动 squeeze 掉 channel 维 (size=1 的维度),保证变成 (N, H, W)
    pixel_label = np.squeeze(pixel_label)
    pixel_preds = np.squeeze(pixel_preds)
    if pixel_label.ndim != 3 or pixel_preds.ndim != 3:
        raise ValueError(
            f"After squeeze, expected 3D arrays for pixel_label/pixel_preds, got {pixel_label.shape}, {pixel_preds.shape}"
        )
    # 形状必须一致
    if pixel_label.shape != pixel_preds.shape:
        raise ValueError(
            f"Shape mismatch: pixel_label {pixel_label.shape} vs pixel_preds {pixel_preds.shape}"
        )

    # 全局背景作为负样本池:从所有图的 GT 取并集的补集
    all_gt = (pixel_label > 0.5).astype(np.uint8)
    all_pred = pixel_preds

    # 用于在 FPR<=max_fpr 区间内插值评估
    n_thresholds = 200
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)

    region_aucs = []
    # 逐图统计连通域
    for i in range(all_gt.shape[0]):
        gt_i = all_gt[i]
        pred_i = all_pred[i]
        labeled, num_cc = cc_label(gt_i)
        if num_cc == 0:
            continue
        # 当前图的背景像素 (作为该图所有区域的共享负样本)
        bg_mask = (labeled == 0)
        bg_pred = pred_i[bg_mask]
        n_bg = bg_pred.size
        if n_bg == 0:
            continue

        for cc_idx in range(1, num_cc + 1):
            region_mask = (labeled == cc_idx)
            region_pred = pred_i[region_mask]
            n_region = region_pred.size
            if n_region == 0:
                continue

            # 在不同阈值下计算 per-region TPR 与 FPR
            tpr_list = []
            fpr_list = []
            for th in thresholds:
                # 正样本 (该区域) 中被预测为异常的比例
                tpr = np.mean(region_pred >= th)
                # 负样本 (背景) 中被预测为异常的比例
                fpr = np.mean(bg_pred >= th)
                tpr_list.append(tpr)
                fpr_list.append(fpr)

            tpr_arr = np.array(tpr_list)
            fpr_arr = np.array(fpr_list)
            # 单调化 (按 fpr 排序)
            order = np.argsort(fpr_arr)
            fpr_sorted = fpr_arr[order]
            tpr_sorted = tpr_arr[order]
            # 只保留 fpr <= max_fpr 区间,通过线性插值到 [0, max_fpr] 求面积
            valid = fpr_sorted <= max_fpr
            if valid.sum() < 2:
                region_aucs.append(0.0)
                continue
            fpr_clip = fpr_sorted[valid]
            tpr_clip = tpr_sorted[valid]
            # 在 [0, max_fpr] 上重采样为均匀网格
            fpr_grid = np.linspace(0.0, max_fpr, 100)
            tpr_grid = np.interp(fpr_grid, fpr_clip, tpr_clip)
            # 归一化到 [0,1] (除以 max_fpr)
            region_auc = np.trapz(tpr_grid, fpr_grid) / max_fpr
            region_aucs.append(region_auc)

    if len(region_aucs) == 0:
        return 0.0
    return float(np.mean(region_aucs))


def calculate_best_f1(pixel_label: np.ndarray, pixel_preds: np.ndarray):
    """
    基于 precision-recall 曲线计算像素级 best F1 score (在最优阈值下)。
    参数:
        pixel_label: shape (N, H, W) 或 (N, 1, H, W), 二值 ground truth (0/1)
        pixel_preds: 同上
    返回:
        best_f1: 最佳 F1 分数 (0~1)
        best_th:  对应阈值
    """
    # 自动 squeeze 掉 channel 维,保证两边形状一致
    pixel_label = np.squeeze(pixel_label)
    pixel_preds = np.squeeze(pixel_preds)
    if pixel_label.shape != pixel_preds.shape:
        raise ValueError(
            f"Shape mismatch: pixel_label {pixel_label.shape} vs pixel_preds {pixel_preds.shape}"
        )
    gt_flat = pixel_label.flatten().astype(np.int32)
    pr_flat = pixel_preds.flatten().astype(np.float64)

    if gt_flat.sum() == 0 or gt_flat.sum() == gt_flat.size:
        return 0.0, 0.5

    precisions, recalls, thresholds = precision_recall_curve(gt_flat, pr_flat)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    # precision_recall_curve 返回的 thresholds 长度比 precisions 少 1
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_f1, best_th


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    # Min-Max Normalization
    if pixel_preds.max() > 1 or pixel_preds.min() < 0:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() > 1 or image_preds.min() < 0:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )

    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = pmax_pred * 0.5 + image_preds * 0.5
    else:
        image_preds = pmax_pred

    # ================================================================================================
    # pixel level metrics (Classification)
    pixel_label_flat = pixel_label.flatten()
    pixel_preds_flat = pixel_preds.flatten()

    pixel_auc = roc_auc_score(pixel_label_flat, pixel_preds_flat)
    pixel_aupr = average_precision_score(pixel_label_flat, pixel_preds_flat)

    # pixel-level AUPRO (per-region overlap) 与 best F1
    pixel_aupro = calculate_aupro(pixel_label, pixel_preds, max_fpr=0.3)
    pixel_best_f1, _ = calculate_best_f1(pixel_label, pixel_preds)

    # ================================================================================================
    # image level metrics (Classification)
    if image_label.max() != image_label.min():
        image_label_flat = image_label.flatten()
        image_preds_flat = image_preds.flatten()

        image_auc = roc_auc_score(image_label_flat, image_preds_flat)
        image_aupr = average_precision_score(image_label_flat, image_preds_flat)

        # image-level best F1
        img_best_f1, _ = calculate_best_f1(image_label, image_preds)
    else:
        image_auc = 0.0
        image_aupr = 0.0
        img_best_f1 = 0.0

    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(pixel_auc, 4) * 100,           # 即 AUROC (pixel)
        "pixel AP": round(pixel_aupr, 4) * 100,
        "pixel AUPRO": round(pixel_aupro, 4) * 100,
        "pixel F1": round(pixel_best_f1, 4) * 100,
        "image AUC": round(image_auc, 4) * 100,           # 即 AUROC (image)
        "image AP": round(image_aupr, 4) * 100,
        "image F1": round(img_best_f1, 4) * 100,
    }

    return result

def extract_variant_from_path(path: str) -> str:
    parts = path.strip().split('/')
    return parts[-1] if len(parts) == 1 else parts[-1] if '.png' in parts[-1] and len(parts) == 1 else parts[0] if parts[0] else parts[1] if parts[0]=='' else parts[0]

def group_indices_by_variant(file_names: list) -> dict:
    variant_map = {}
    for idx, fn in enumerate(file_names):
        variant = extract_variant_from_path(fn)
        variant_map.setdefault(variant, []).append(idx)
    return variant_map

def normalize(pred):
    if (pred.max() - pred.min()) == 0:
        return np.zeros_like(pred)
    return (pred - pred.min()) / (pred.max() - pred.min())


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)

    scoremap = np.squeeze(scoremap)
    if scoremap.ndim != 2:
        raise ValueError(f"Expected 2D scoremap, got shape {scoremap.shape}")

    scoremap = normalize(scoremap) 
    scoremap = (scoremap * 255).astype(np.uint8)

    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)

    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def he_cheng(img_list, size=256):
    h, w, c = img_list[0].shape
    pad = np.ones((h, 10, 3), dtype=np.uint8) * 255
    vis = img_list[0]
    for i in range(1, len(img_list)):
        vis = np.concatenate([vis, pad, img_list[i]], axis=1)
    vis = cv2.resize(vis, (size * len(img_list) + 10 * (len(img_list) - 1), size))
    return vis.astype(np.uint8)


def visualization(save_root, pic_name, raw_image, raw_anomaly_map, raw_gt, the=0.5, size=518):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    raw_gt = np.squeeze(raw_gt)
    raw_anomaly_map = np.squeeze(raw_anomaly_map)
    
    if raw_anomaly_map.ndim > 2:
        raw_anomaly_map = raw_anomaly_map[0]
    if raw_gt.ndim > 2:
        raw_gt = raw_gt[0]
    
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    map_norm = normalize(raw_anomaly_map)
    gt_norm = normalize(raw_gt)

    map_binary = np.array(raw_anomaly_map > the, dtype=np.uint8)
    map_binary = np.squeeze(map_binary)
    map_crop = map_norm * map_binary

    ground_truth_contours, _ = cv2.findContours(np.array(raw_gt * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis_map = apply_ad_scoremap(img, map_norm)
    vis_gt = apply_ad_scoremap(img, gt_norm)
    vis_map_binary = apply_ad_scoremap(img, map_binary)
    vis_map_crop = apply_ad_scoremap(img, map_crop)

    vis_map = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR)
    vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
    vis_map_binary = cv2.cvtColor(vis_map_binary, cv2.COLOR_RGB2BGR)
    vis_map_crop = cv2.cvtColor(vis_map_crop, cv2.COLOR_RGB2BGR)

    if len(ground_truth_contours) > 0:
        vis_map_binary = cv2.drawContours(vis_map_binary, ground_truth_contours, -1, (0, 255, 0), 2)

    merged_img = he_cheng([raw_image, vis_map, vis_map_crop, vis_gt])
    save_path = os.path.join(save_root, pic_name.replace('bmp', 'png').replace('jpg', 'png'))
    cv2.imwrite(save_path, merged_img)

def calcuate_metric_pixel(results, obj_list, args, sigma=8):

    for obj in obj_list:
        gt_px = results['imgs_masks']
        pr_px = results["anomaly_maps"]
        paths = results["path"]

        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)

        if pr_px.size == 0 or gt_px.size == 0:
            print(f"[WARN] No samples for class '{obj}', skipping.")
            continue

        if sigma != 0:
            pr_px = gaussian_filter(pr_px, sigma=sigma, axes=(1, 2))

        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_th = thresholds[np.argmax(f1_scores)]
        f1_best = np.max(f1_scores)
        #print(f"[{obj}] threshold={best_th:.4f}, F1_best={f1_best:.4f}, mean_pred={pr_px.mean():.4f}")

        gt_px = gt_px.squeeze()
        pr_px = pr_px.squeeze()

        for i in range(len(paths)):
            #print("len path: ", len(paths))
            img = cv2.resize(cv2.imread(paths[i]), (args.img_size, args.img_size))

            anomaly_type = os.path.normpath(paths[i]).split(os.sep)[-2]
            
            save_dir = os.path.join(args.visual_path, "visualization", args.dataset, obj, anomaly_type)
            os.makedirs(save_dir, exist_ok=True)

            pic_name = os.path.basename(paths[i])

            visualization(
                save_root=save_dir,
                pic_name=pic_name,
                raw_image=img,
                raw_anomaly_map=np.squeeze(pr_px[i]),
                raw_gt=np.squeeze(gt_px[i]),
                the=best_th,
             
                size=args.img_size,
            )