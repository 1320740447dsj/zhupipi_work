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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim
from scipy.ndimage import label as cc_label

# ================================================================================================
# The following code is used to get criterion for training


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


def calculate_aupro(pixel_label, pixel_preds, max_fpr=0.3):
    """
    计算 AUPRO (Area Under the Per-Region Overlap)。
    参考 MVTec AD 标准实现：对 GT mask 的每个连通区域计算 TPR，
    对所有非异常像素计算 FPR，最终取 FPR ∈ [0, max_fpr] 区间内的 AUC 并归一化。

    参数：
        pixel_label: (N, H, W) 二值 GT mask (0/1)
        pixel_preds: (N, H, W) 预测分数 (已归一化到 [0,1])
        max_fpr: PRO 曲线 FPR 上限，默认 0.3 (MVTec 标准)
    返回：
        aupro: 标量，0~1 之间
    """
    # 把多张图当作一张大图来算（与 MVTec 一致）
    preds_flat = pixel_preds.flatten()
    label_flat = pixel_label.flatten()

    # 没有正样本直接返回 0
    if label_flat.sum() == 0:
        return 0.0

    # 共享阈值集合：用预测值的去重分位作为候选阈值
    # 数量过大会非常慢，这里采样 300 个阈值
    n_thresh = 300
    if len(preds_flat) > n_thresh:
        thresholds = np.quantile(preds_flat, np.linspace(0, 1, n_thresh))
    else:
        thresholds = np.unique(preds_flat)

    pro_scores = []  # 每个 region 的 TPR
    fpr_list = []    # 每个 threshold 下的 (mean region TPR, FPR)

    for thresh in thresholds:
        binary_pred = (preds_flat >= thresh).astype(np.uint8)
        # ---- per-region overlap (TPR per connected component) ----
        # 逐样本找连通区域，避免跨样本合并
        per_region_tpr = []
        n_total_regions = 0
        for i in range(pixel_label.shape[0]):
            gt = pixel_label[i].astype(np.uint8)
            if gt.sum() == 0:
                continue
            labeled, n_cc = cc_label(gt)
            pred_i = binary_pred.reshape(pixel_label.shape)[i]
            for cc_id in range(1, n_cc + 1):
                region_mask = (labeled == cc_id)
                region_size = region_mask.sum()
                if region_size == 0:
                    continue
                overlap = (region_mask & (pred_i > 0)).sum()
                per_region_tpr.append(overlap / region_size)
                n_total_regions += 1
        if n_total_regions == 0:
            mean_tpr = 0.0
        else:
            mean_tpr = float(np.mean(per_region_tpr))
        # ---- FPR on normal pixels ----
        normal_mask = (label_flat == 0)
        if normal_mask.sum() == 0:
            fpr = 0.0
        else:
            fpr = float(binary_pred[normal_mask].sum() / normal_mask.sum())
        fpr_list.append((fpr, mean_tpr))

    # 按 FPR 排序构造 PRO 曲线
    fpr_list = np.array(sorted(fpr_list, key=lambda x: x[0]))
    fprs = fpr_list[:, 0]
    tprs = fpr_list[:, 1]

    # 只取 FPR <= max_fpr 的部分
    valid = fprs <= max_fpr
    if valid.sum() < 2:
        return 0.0
    # 在 [0, max_fpr] 上插入端点保证积分区间完整
    fprs_v = np.concatenate([[0.0], fprs[valid], [max_fpr]])
    tprs_v = np.concatenate([[0.0], tprs[valid], [tprs[valid][-1]]])
    # np.trapz 返回的面积归一化到 [0, max_fpr]
    aupro = float(np.trapz(tprs_v, fprs_v) / max_fpr)
    return aupro


def calculate_optimal_f1(pixel_label, pixel_preds):
    """
    在像素级上计算最优阈值下的 F1 score。
    通过 precision-recall curve 找到使 F1 最大的阈值。
    """
    label_flat = pixel_label.flatten().astype(np.uint8)
    preds_flat = pixel_preds.flatten()
    if label_flat.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(label_flat, preds_flat)
    # f1 = 2 * P * R / (P + R)
    denom = (precision + recall)
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-8), 0.0)
    return float(np.max(f1))


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
    if image_preds.max() != 1:
        image_preds = (image_preds - image_preds.min()) / (
            image_preds.max() - image_preds.min()
        )

    pmax_pred = pixel_preds.max(axis=(1, 2))
    if domain != "Medical":
        image_preds = pmax_pred * 0.5 + image_preds * 0.5
    else:
        image_preds = pmax_pred
    # ================================================================================================
    # pixel level auc & ap & aupro & f1
    # 保留 (N, H, W) 形状用于 AUPRO，flatten 用于 AUC/AP
    pixel_label_flat = pixel_label.flatten()
    pixel_preds_flat = pixel_preds.flatten()

    zero_pixel_auc = roc_auc_score(pixel_label_flat, pixel_preds_flat)
    zero_pixel_ap = average_precision_score(pixel_label_flat, pixel_preds_flat)
    # AUPRO: 需要 (N, H, W) 形状的输入
    aupro = calculate_aupro(pixel_label, pixel_preds)
    # F1: 最优阈值下的 pixel-level F1
    pixel_f1 = calculate_optimal_f1(pixel_label, pixel_preds)
    # ================================================================================================
    # image level auc & ap
    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = roc_auc_score(image_label, agg_image_preds)
        agg_image_ap = average_precision_score(image_label, agg_image_preds)
        # image-level F1: 同样用最优阈值
        image_f1 = calculate_optimal_f1(
            image_label.reshape(-1, 1), agg_image_preds.reshape(-1, 1)
        )
    else:
        agg_image_auc = 0
        agg_image_ap = 0
        image_f1 = 0
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc, 4) * 100,
        "pixel AP": round(zero_pixel_ap, 4) * 100,
        "pixel AUPRO": round(aupro, 4) * 100,
        "pixel F1": round(pixel_f1, 4) * 100,
        "image AUC": round(agg_image_auc, 4) * 100,
        "image AP": round(agg_image_ap, 4) * 100,
        "image F1": round(image_f1, 4) * 100,
    }
    return result


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    file_names: list[str],
    save_dir: str,
    dataset_name: str,
    class_name: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
        pixel_preds = (pixel_preds * 255).astype(np.uint8)
    if pixel_label.dtype != np.uint8:
        pixel_label = pixel_label != 0
        pixel_label = (pixel_label * 255).astype(np.uint8)
    # ===============================================================================================
    # save path
    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for idx, file in enumerate(file_names):
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, pixel_label.shape[-2:])
        save_image_list = [image]

        if dataset_name == "MVTec":
            damage_name, image_name = file.split("/")[-2:]
            file_name = f"{damage_name}_{image_name}"
        else:
            raise NotImplementedError

        save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
        save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
        save_image_list = save_image_list[:1] + [
            apply_ad_scoremap(image, _) for _ in save_image_list[1:]
        ]
        scoremap = np.vstack(save_image_list)
        cv2.imwrite(os.path.join(save_dir, file_name), scoremap)
