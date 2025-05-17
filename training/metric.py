import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull

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

def giou(inputs, targets, num_objects):
    """
    计算 GIoU
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）
    :return: GIoU 值
    """
    giou_list = []
    for i in range(1, num_objects + 1):  # 假设背景是 0，从 1 开始计算各个对象的 GIoU
        pred_mask = (inputs == i)
        true_mask = (targets == i)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        # 如果 union 为 0，直接返回 1.0（特殊情况处理）
        if union == 0:
            giou_list.append(1.0)
            continue

        pred_coords = np.argwhere(pred_mask)
        true_coords = np.argwhere(true_mask)
        all_coords = np.concatenate([pred_coords, true_coords], axis=0)
        if len(all_coords) == 0:
            giou_list.append(1.0)
            continue

        # 使用凸包代替目标检测中的最小外接矩形
        hull = ConvexHull(all_coords[:, 1:])
        hull_area = hull.volume  # 2D convex hull volume is the area
        
        # 计算 GIoU
        giou = intersection / union - (hull_area - union) / hull_area
        giou_list.append(giou)
    
    return np.mean(giou_list)


def ciou(inputs, targets, num_objects):
    """
    计算 CIoU
    :param inputs: 预测的分割结果，形状为 [H, W]
    :param targets: 真实的分割标签，形状为 [H, W]
    :param num_objects: mask 个数（类别数）
    :return: CIoU 值
    """
    ciou_list = []
    for i in range(1, num_objects + 1):
        pred_mask = (inputs == i)
        true_mask = (targets == i)
        
        # I和U
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            ciou_list.append(1.0)
            continue
        
        pred_coords = np.argwhere(pred_mask)
        true_coords = np.argwhere(true_mask)
        
        if len(pred_coords) == 0 or len(true_coords) == 0:
            ciou_list.append(0.0)
            continue
        
        # 外接矩形
        x_coords = np.concatenate([pred_coords[:, 1], true_coords[:, 1]])
        y_coords = np.concatenate([pred_coords[:, 2], true_coords[:, 2]])
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        # 边界框
        pred_bbox = np.array([[np.min(pred_coords[:, 1]), np.min(pred_coords[:, 2])],
                              [np.max(pred_coords[:, 1]), np.max(pred_coords[:, 2])]])
        true_bbox = np.array([[np.min(true_coords[:, 1]), np.min(true_coords[:, 2])],
                              [np.max(true_coords[:, 1]), np.max(true_coords[:, 2])]])
        # 边界框宽/高
        w_pred = pred_bbox[1, 0] - pred_bbox[0, 0] + 1
        h_pred = pred_bbox[1, 1] - pred_bbox[0, 1] + 1
        w_true = true_bbox[1, 0] - true_bbox[0, 0] + 1
        h_true = true_bbox[1, 1] - true_bbox[0, 1] + 1
        # 两框的中心距离
        center_distance = np.sqrt(((pred_bbox[0, 0] + pred_bbox[1, 0]) / 2 - (true_bbox[0, 0] + true_bbox[1, 0]) / 2) ** 2 +
                                  ((pred_bbox[0, 1] + pred_bbox[1, 1]) / 2 - (true_bbox[0, 1] + true_bbox[1, 1]) / 2) ** 2)
        # 外接矩形面积
        c = np.sqrt(((max_x - min_x) ** 2) + ((max_y - min_y) ** 2))
        
        v = (4 / (np.pi ** 2)) * (np.arctan(w_true / h_true) - np.arctan(w_pred / h_pred)) ** 2
        
        alpha = v / (1 - intersection / union + v)
        
        ciou = intersection / union - (center_distance ** 2) / c ** 2 - alpha * v
        ciou_list.append(ciou)
    
    return np.mean(ciou_list)

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

def estimate(inputs_list, targets_list, num_objects, metric_dict):
    for inputs, targets in zip(inputs_list, targets_list):
        for (metric_name, metric_func), score_list in metric_dict.items():
            score_list.append(metric_func(inputs, targets, num_objects))
    return metric_dict

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

    metric_dict = {("IoU", iou): [], ("GIoU", giou): [], ("CIoU", ciou): [], ("Dice", dice): [], ("Pixel Accuracy", pixel_accuracy): [], ("Boundary F1 Score", boundary_f1_score): []}
    num_objects = 1
    metric_dict = estimate(inputs_list, targets_list, num_objects, metric_dict)

    for (metric_name, _), score_list in metric_dict.items():
        print(f"Average {metric_name}:", np.mean(score_list))