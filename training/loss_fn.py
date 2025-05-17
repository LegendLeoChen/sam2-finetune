from collections import defaultdict
from typing import Dict, List, Tuple

import time
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
import numpy as np

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean((1, 2, 3)).sum() / num_objects


def diou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    交并比损失
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted dIoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        dIoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def iou_loss(
    inputs, targets, num_objects, loss_on_multimask=False
):
    """
    交并比损失
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    loss = 1 - torch.tanh(actual_ious * 3)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def giou_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    GIoU 损失函数
    Args:
        inputs: 预测的分割结果，形状为 [bs, n, H, W]
        targets: 真实的分割标签，形状为 [bs, n, H, W]
        num_objects: mask 个数（类别数）
        loss_on_multimask: 是否对每个 mask 单独计算损失
    Returns:
        GIoU 损失值
    """
    bs, n, H, W = inputs.shape

    # 提取预测和真实的目标 mask
    pred_mask = inputs > 0  # [bs, n, H, W]
    gt_mask = targets > 0   # [bs, n, H, W]

    # 计算交集和并集
    intersection = torch.sum(pred_mask & gt_mask, dim=(2, 3)).float()  # [bs, n]
    union = torch.sum(pred_mask | gt_mask, dim=(2, 3)).float()  # [bs, n]

    # 计算边界框
    # 初始化边界框坐标
    pred_bbox = torch.zeros((bs, n, 4), device=inputs.device)  # [bs, n, 4] (x1, y1, x2, y2)
    gt_bbox = torch.zeros((bs, n, 4), device=inputs.device)  # [bs, n, 4] (x1, y1, x2, y2)

    # 获取预测和真实目标的非零坐标
    pred_coords = torch.nonzero(pred_mask, as_tuple=False)  # [num_points, 4], (bs, n, H, W)
    gt_coords = torch.nonzero(gt_mask, as_tuple=False)  # [num_points, 4], (bs, n, H, W)

    # 如果没有非零点，直接返回损失为0
    if pred_coords.numel() == 0 or gt_coords.numel() == 0:
        return torch.zeros(1, device=inputs.device)

    # 计算预测边界框的坐标
    pred_min_coords = torch.min(pred_coords[:, 2:], dim=0).values  # [2] (min_H, min_W)
    pred_max_coords = torch.max(pred_coords[:, 2:], dim=0).values  # [2] (max_H, max_W)
    pred_bbox[:, :, 0] = pred_min_coords[1]  # x1
    pred_bbox[:, :, 1] = pred_min_coords[0]  # y1
    pred_bbox[:, :, 2] = pred_max_coords[1]  # x2
    pred_bbox[:, :, 3] = pred_max_coords[0]  # y2

    # 计算真实边界框的坐标
    gt_min_coords = torch.min(gt_coords[:, 2:], dim=0).values  # [2] (min_H, min_W)
    gt_max_coords = torch.max(gt_coords[:, 2:], dim=0).values  # [2] (max_H, max_W)
    gt_bbox[:, :, 0] = gt_min_coords[1]  # x1
    gt_bbox[:, :, 1] = gt_min_coords[0]  # y1
    gt_bbox[:, :, 2] = gt_max_coords[1]  # x2
    gt_bbox[:, :, 3] = gt_max_coords[0]  # y2

    # 计算边界框的交集和并集
    x1 = torch.max(pred_bbox[:, :, 0], gt_bbox[:, :, 0])  # [bs, n]
    y1 = torch.max(pred_bbox[:, :, 1], gt_bbox[:, :, 1])  # [bs, n]
    x2 = torch.min(pred_bbox[:, :, 2], gt_bbox[:, :, 2])  # [bs, n]
    y2 = torch.min(pred_bbox[:, :, 3], gt_bbox[:, :, 3])  # [bs, n]

    # 计算交集面积
    intersection_bbox = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)  # [bs, n]

    # 计算并集面积
    pred_area = (pred_bbox[:, :, 2] - pred_bbox[:, :, 0]) * (pred_bbox[:, :, 3] - pred_bbox[:, :, 1])  # [bs, n]
    gt_area = (gt_bbox[:, :, 2] - gt_bbox[:, :, 0]) * (gt_bbox[:, :, 3] - gt_bbox[:, :, 1])  # [bs, n]
    union_bbox = pred_area + gt_area - intersection_bbox  # [bs, n]

    # 计算 GIoU 损失
    loss = 1 - (intersection / union - (union_bbox - union) / union_bbox)  # [bs, n]

    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / (bs * num_objects)

def total_loss(
    inputs,
    targets,
    pred_ious,
    num_objects,
    alpha: float = 0.3,
    gamma: float = 2,
    loss_on_multimask: bool = False,
    use_l1_loss: bool = False,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    diou_weight: float = 1.0,
    iou_weight: float = 1.0,
    giou_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute the total loss by combining Dice loss, Focal loss, and IoU loss.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                 (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoU scores per mask.
        num_objects: Number of objects in the batch.
        alpha: (optional) Weighting factor in range (0,1) to balance
               positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Default = 2.
        loss_on_multimask: True if multimask prediction is enabled.
        use_l1_loss: Whether to use L1 loss instead of MSE loss for IoU loss.
        dice_weight: Weight for the Dice loss component.
        focal_weight: Weight for the Focal loss component.
        diou_weight: Weight for the dIoU loss component.
        iou_weight: Weight for the IoU loss component.

    Returns:
        Total loss tensor.
    """
    # Compute individual losses
    dice = dice_loss(inputs, targets, num_objects, loss_on_multimask)
    focal = sigmoid_focal_loss(inputs, targets, num_objects, alpha, gamma, loss_on_multimask)
    diou = diou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask, use_l1_loss)
    iou = iou_loss(inputs, targets, num_objects, loss_on_multimask)
    # giou = giou_loss(inputs, targets, num_objects, loss_on_multimask)

    # Combine losses with weights
    total = dice_weight * dice + focal_weight * focal + diou_weight * diou + iou * iou_weight

    return total, {"dice": dice_weight * dice, "focal": focal_weight * focal, "diou": diou_weight * diou, "iou": iou_weight * iou}