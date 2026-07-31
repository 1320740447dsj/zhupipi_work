"""Disk-friendly streaming metrics for large anomaly detection datasets."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass
class _ClassState:
    pixel_positive: torch.Tensor
    pixel_negative: torch.Tensor
    region_overlap: torch.Tensor
    region_count: int = 0
    image_targets: list = field(default_factory=list)
    image_scores: list = field(default_factory=list)


class StreamingEvaluator:
    """Accumulate threshold-binned metrics without retaining prediction maps."""

    def __init__(self, device, metrics, num_thresholds=512, fpr_limit=0.3):
        if num_thresholds < 2:
            raise ValueError('num_thresholds must be at least 2')
        self.device = torch.device(device)
        self.metrics = metrics
        self.num_thresholds = num_thresholds
        self.fpr_limit = fpr_limit
        self.states = {}

    def _new_state(self):
        zeros = lambda: torch.zeros(self.num_thresholds, dtype=torch.float64, device=self.device)
        return _ClassState(zeros(), zeros(), zeros())

    def update(self, class_name, pixel_scores, pixel_targets, image_scores, image_targets):
        state = self.states.setdefault(class_name, self._new_state())
        scores = pixel_scores.detach().to(self.device, dtype=torch.float32).clamp_(0, 1)
        targets = pixel_targets.detach().to(self.device).bool()
        bins = torch.clamp((scores * (self.num_thresholds - 1)).long(), 0, self.num_thresholds - 1)

        state.pixel_positive += torch.bincount(
            bins[targets], minlength=self.num_thresholds
        ).to(torch.float64)
        state.pixel_negative += torch.bincount(
            bins[~targets], minlength=self.num_thresholds
        ).to(torch.float64)

        targets_cpu = targets.cpu().numpy().astype(np.uint8)
        for index, target in enumerate(targets_cpu):
            component_count, components = cv2.connectedComponents(target, connectivity=8)
            if component_count <= 1:
                continue
            areas = np.bincount(components.ravel(), minlength=component_count)
            weights = np.zeros_like(components, dtype=np.float32)
            foreground = components > 0
            weights[foreground] = 1.0 / areas[components[foreground]]
            weights = torch.from_numpy(weights).to(self.device)
            state.region_overlap += torch.bincount(
                bins[index][targets[index]],
                weights=weights[targets[index]],
                minlength=self.num_thresholds,
            ).to(torch.float64)
            state.region_count += component_count - 1

        state.image_scores.extend(image_scores.detach().float().cpu().tolist())
        state.image_targets.extend(image_targets.detach().int().cpu().tolist())

    @staticmethod
    def _reverse_cumulative(histogram):
        return torch.flip(torch.cumsum(torch.flip(histogram, dims=[0]), dim=0), dims=[0])

    @staticmethod
    def _binary_f1(tp, fp, positive_count):
        fn = positive_count - tp
        return 2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)

    def _pixel_curves(self, state):
        tp = self._reverse_cumulative(state.pixel_positive)
        fp = self._reverse_cumulative(state.pixel_negative)
        positives = state.pixel_positive.sum()
        negatives = state.pixel_negative.sum()
        tpr = tp / positives.clamp_min(1)
        fpr = fp / negatives.clamp_min(1)
        return tp, fp, positives, torch.flip(fpr, dims=[0]), torch.flip(tpr, dims=[0])

    def _pixel_auroc(self, state):
        _, _, _, fpr, tpr = self._pixel_curves(state)
        zero = torch.zeros(1, dtype=fpr.dtype, device=fpr.device)
        fpr = torch.cat([zero, fpr])
        tpr = torch.cat([zero, tpr])
        return torch.trapz(tpr, fpr).item()

    def _pixel_f1(self, state):
        tp, fp, positives, _, _ = self._pixel_curves(state)
        return self._binary_f1(tp, fp, positives).max().item()

    def _pixel_ap(self, state):
        tp, fp, positives, _, _ = self._pixel_curves(state)
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / positives.clamp_min(1)
        precision = torch.flip(precision, dims=[0])
        recall = torch.flip(recall, dims=[0])
        recall_delta = recall[1:] - recall[:-1]
        return torch.sum(recall_delta * precision[1:]).item()

    def _aupro(self, state):
        if state.region_count == 0:
            return float('nan')
        false_positives = self._reverse_cumulative(state.pixel_negative)
        overlap = self._reverse_cumulative(state.region_overlap) / state.region_count
        fpr = false_positives / state.pixel_negative.sum().clamp_min(1)
        fpr = torch.flip(fpr, dims=[0])
        overlap = torch.flip(overlap, dims=[0])
        zero = torch.zeros(1, dtype=fpr.dtype, device=fpr.device)
        fpr = torch.cat([zero, fpr])
        overlap = torch.cat([zero, overlap])

        below = fpr <= self.fpr_limit
        x = fpr[below]
        y = overlap[below]
        above_indices = torch.where(fpr > self.fpr_limit)[0]
        if above_indices.numel() and x[-1] < self.fpr_limit:
            upper = above_indices[0]
            lower = upper - 1
            ratio = (self.fpr_limit - fpr[lower]) / (fpr[upper] - fpr[lower]).clamp_min(1e-12)
            y_limit = overlap[lower] + ratio * (overlap[upper] - overlap[lower])
            x = torch.cat([x, x.new_tensor([self.fpr_limit])])
            y = torch.cat([y, y_limit.unsqueeze(0)])
        return (torch.trapz(y, x) / self.fpr_limit).item()

    def compute(self, class_name):
        state = self.states[class_name]
        image_targets = np.asarray(state.image_targets)
        image_scores = np.asarray(state.image_scores)
        results = {}
        for metric in self.metrics:
            if metric.startswith('I-AUROC'):
                results[metric] = roc_auc_score(image_targets, image_scores)
            elif metric.startswith('I-AP'):
                results[metric] = average_precision_score(image_targets, image_scores)
            elif metric.startswith('I-F1max'):
                precision, recall, _ = precision_recall_curve(image_targets, image_scores)
                results[metric] = np.nanmax(2 * precision * recall / (precision + recall + 1e-12))
            elif metric.startswith('P-AUROC'):
                results[metric] = self._pixel_auroc(state)
            elif metric.startswith('P-AP'):
                results[metric] = self._pixel_ap(state)
            elif metric.startswith('P-F1max'):
                results[metric] = self._pixel_f1(state)
            elif metric.startswith('P-AUPRO'):
                results[metric] = self._aupro(state)
            else:
                raise ValueError(f'Unsupported streaming metric: {metric}')
        return results
