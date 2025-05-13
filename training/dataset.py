import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os


# 定义 Dataset 类
class LabPicsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        image_folder = os.path.join(self.data_dir, "Image")
        annotation_folder = os.path.join(self.data_dir, "Instance")
        for name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, name)
            annotation_path = os.path.join(annotation_folder, name[:-4] + ".png")
            data.append({"image": image_path, "annotation": annotation_path})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = cv2.imread(entry["image"])[..., ::-1]  # 读取图像并转换为 RGB 格式
        annotation = cv2.imread(entry["annotation"])  # 读取标注

        # 调整图像大小
        r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])  # 缩放因子
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
        annotation = cv2.resize(annotation, (int(annotation.shape[1] * r), int(annotation.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        if image.shape[0] < 1024:
            image = np.concatenate([image, np.zeros([1024 - image.shape[0], image.shape[1], 3], dtype=np.uint8)], axis=0)
            annotation = np.concatenate([annotation, np.zeros([1024 - annotation.shape[0], annotation.shape[1], 3], dtype=np.uint8)], axis=0)
        if image.shape[1] < 1024:
            image = np.concatenate([image, np.zeros([image.shape[0], 1024 - image.shape[1], 3], dtype=np.uint8)], axis=1)
            annotation = np.concatenate([annotation, np.zeros([annotation.shape[0], 1024 - annotation.shape[1], 3], dtype=np.uint8)], axis=1)

        # 合并材料和容器标注
        mat_map = annotation[:, :, 0]
        ves_map = annotation[:, :, 2]
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

        # 获取二值掩码和点
        inds = np.unique(mat_map)[1:]
        if len(inds) > 0:
            ind = inds[np.random.randint(len(inds))]
            # mask = (mat_map == ind).astype(np.uint8)
            mask = (mat_map > 0).astype(np.uint8)           # 全部mask合成一个（全部预测出来）
            coords = np.argwhere(mask > 0)
            yx = coords[np.random.randint(len(coords))]
            point = [[yx[1], yx[0]]]
        else:
            # 如果没有有效标注，返回全零掩码和随机点
            mask = np.zeros((image.shape[:2]), dtype=np.uint8)
            point = [[np.random.randint(0, 1024), np.random.randint(0, 1024)]]

        if self.transform:
            image = self.transform(image)

        return image, mask, np.array(point, dtype=np.float32), np.ones([1])
