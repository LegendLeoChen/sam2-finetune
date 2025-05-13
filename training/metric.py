import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

def iou(inputs, targets, num_objects):
    """
    计算交并比（IoU）
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）
    :return: IoU 值
    """
    iou_list = []
    for i in range(1, num_objects + 1):  # 假设背景是 0，从 1 开始计算各个对象的 IoU
        pred_mask = (inputs == i)
        true_mask = (targets == i)
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            iou_list.append(1.0)  # 如果 union 为 0，IoU 定义为 1（特殊情况处理）
        else:
            iou_list.append(intersection / union)
    return np.mean(iou_list)

def dice(inputs, targets, num_objects):
    """
    计算 Dice 系数
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）
    :return: Dice 系数
    """
    dice_list = []
    for i in range(1, num_objects + 1):  # 假设背景是 0，从 1 开始计算各个对象的 Dice
        pred_mask = (inputs == i)
        true_mask = (targets == i)
        intersection = np.logical_and(pred_mask, true_mask).sum()
        dice_value = (2. * intersection) / (pred_mask.sum() + true_mask.sum())
        dice_list.append(dice_value)
    return np.mean(dice_list)

def pixel_accuracy(inputs, targets, num_objects):
    """
    计算像素准确度
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）（虽然这里用不到，但保持接口一致）
    :return: 像素准确度
    """
    correct = (inputs == targets).sum()
    total = inputs.size
    return correct / total

def boundary_f1_score(inputs, targets, num_objects):
    """
    计算边界 F1 分数
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）
    :return: 边界 F1 分数
    """
    # 创建一个结构元素用于边界检测（这里使用 3x3 的十字结构）
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    f1_list = []
    for i in range(1, num_objects + 1):  # 对每个对象计算边界 F1 分数
        # 获取预测和真实的目标掩码
        pred_mask = (inputs == i).astype(np.uint8)[0]
        true_mask = (targets == i).astype(np.uint8)[0]

        # 计算边界
        pred_boundary = binary_dilation(pred_mask, struct) ^ binary_erosion(pred_mask, struct)
        true_boundary = binary_dilation(true_mask, struct) ^ binary_erosion(true_mask, struct)

        # 计算 TP、FP、FN
        tp = np.logical_and(pred_boundary, true_boundary).sum()
        fp = np.logical_and(pred_boundary, np.logical_not(true_boundary)).sum()
        fn = np.logical_and(np.logical_not(pred_boundary), true_boundary).sum()

        # 计算精确度和召回率
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        # 计算 F1 分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_list.append(f1)

    return np.mean(f1_list)

# 示例
if __name__ == "__main__":
    # 第一张图片
    inputs1 = np.array([[1, 1, 0],
                        [1, 2, 0],
                        [0, 0, 0]])

    targets1 = np.array([[1, 0, 0],
                         [1, 2, 0],
                         [0, 0, 0]])

    # 第二张图片
    inputs2 = np.array([[2, 1, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    targets2 = np.array([[2, 0, 0],
                         [2, 1, 0],
                         [0, 0, 0]])

    inputs_list = [inputs1, inputs2]
    targets_list = [targets1, targets2]
    num_objects = 2

    iou_scores = []
    dice_scores = []
    pixel_acc_scores = []
    boundary_f1_scores = []

    for inputs, targets in zip(inputs_list, targets_list):
        iou_scores.append(iou(inputs, targets, num_objects))
        dice_scores.append(dice(inputs, targets, num_objects))
        pixel_acc_scores.append(pixel_accuracy(inputs, targets, num_objects))
        boundary_f1_scores.append(boundary_f1_score(inputs, targets, num_objects))

    print("Average IoU:", np.mean(iou_scores))
    print("Average Dice:", np.mean(dice_scores))
    print("Average Pixel Accuracy:", np.mean(pixel_acc_scores))
    print("Average Boundary F1 Score:", np.mean(boundary_f1_scores))