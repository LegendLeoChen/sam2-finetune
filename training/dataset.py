import numpy as np
import cv2
import torch
import json
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import torch.nn.functional as F

from sam2.utils.transforms import SAM2Transforms

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
            
        mask = np.expand_dims(mask, axis=0)
        if self.transform:
            image = self.transform(image)

        return image, mask, np.array(point, dtype=np.float32), np.ones([1])

 
class SemSegDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        image_size: int = 1024,
        num_classes_per_sample: int = 1,
        dataset_type: str = "training",
        transform=None,
    ):
        self.transform = transform
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.dataset_type = dataset_type
        
        classes, images, labels = self.init_ade20k(base_image_dir)
        self.data = (images, labels)
        self.classes = classes

    def init_ade20k(self, base_image_dir):
        with open("./ade20k_classes.json", "r") as f:
            ade20k_classes = json.load(f)
        ade20k_classes = np.array(ade20k_classes)
        image_ids = sorted(
            os.listdir(os.path.join(base_image_dir, "ade20k/images", self.dataset_type))
        )
        ade20k_image_ids = []
        for x in image_ids:
            if x.endswith(".jpg"):
                ade20k_image_ids.append(x[:-4])
        ade20k_images = []
        for image_id in ade20k_image_ids:  # self.descriptions:
            ade20k_images.append(
                os.path.join(
                    base_image_dir,
                    "ade20k",
                    "images",
                    self.dataset_type,
                    "{}.jpg".format(image_id),
                )
            )
        ade20k_labels = [
            x.replace(".jpg", ".png").replace("images", "annotations")
            for x in ade20k_images
        ]
        print(f"ade20k for {self.dataset_type}: {len(ade20k_images)}")
        return ade20k_classes, ade20k_images, ade20k_labels
    
    def __len__(self):
        return len(self.data[0])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image, labels = self.data
        image_path = image[idx]
        label_path = labels[idx]
        
        label = Image.open(label_path)
        label = np.array(label)
        label[label == 0] = 255
        label -= 1
        label[label == 254] = 255

        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     image = self.transform(image)
            
        unique_label = np.unique(label).tolist()            # 按类别分出所有mask
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0:
            return self.__getitem__(0)

        classes = [self.classes[class_id] for class_id in unique_label]     # 图中有的类名
        if len(classes) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(
                classes, size=self.num_classes_per_sample, replace=False
            ).tolist()
        else:
            sampled_classes = classes

        class_ids = []                  # 类别索引
        for sampled_cls in sampled_classes:
            class_id = self.classes.tolist().index(sampled_cls)
            class_ids.append(class_id)

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        label = self.padding(torch.from_numpy(label).long())
        
        masks = []
        points = []
        points_label = []
        for class_id in class_ids:
            mask = label == class_id
            masks.append(mask)
            coords = torch.nonzero(mask, as_tuple=False)            # 坐标点
            if coords.numel() > 0:
                coords = coords.view(-1, 2)
                yx = coords[torch.randint(0, coords.shape[0], (1,)).item()]
                points.append([yx[1].item(), yx[0].item()])
                points_label.append(1)
            else:
                points.append([random.randint(0, 1024), random.randint(0, 1024)])
                points_label.append(0)
        
        masks = torch.stack(masks, dim=0)
        return (
            image,
            masks,
            np.array(points, dtype=np.float32),
            np.array(points_label, dtype=np.float32),
            # label,
            # sampled_classes,
            # image_path
        )