import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.metric import *
from training.dataset import LabPicsDataset, SemSegDataset

os.chdir(sys.path[0])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
model_scale_dict = {"b+": "base_plus", "l": "large", "s": "small", "t": "tiny"}
model_scale = "l"

# 数据集和数据加载器
# dataset = LabPicsDataset("../assets/LabPicsV1/Simple/Test", transform=transforms.ToTensor())
dataset = SemSegDataset("../../datasets", dataset_type="validation", transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 加载模型
sam2_checkpoint = f"../checkpoints/sam2_hiera_{model_scale_dict[model_scale]}.pt"
model_cfg = f"../sam2_configs/sam2_hiera_{model_scale}.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(f"./output/exp{17}/model_epoch_{30}.pt"))

use_prompt = True

metric_dict = {("IoU", iou): [], ("GIoU", giou): [], ("CIoU", ciou): [], ("Dice", dice): [], ("Pixel Accuracy", pixel_accuracy): [], ("Boundary F1 Score", boundary_f1_score): []}
# metric_dict = {("IoU", iou): [], ("Dice", dice): [], ("Pixel Accuracy", pixel_accuracy): []}

num = 0
for batch_idx, (image, mask, input_point, input_label) in enumerate(tqdm(dataloader)):
    # num += 1
    # if num > 100:
    #     break
    gt_mask = mask.float()
    gt_mask = [gt_mask[i].numpy() for i in range(gt_mask.shape[0])]
    predictor.set_image_batch(image)
    masks, scores, logits = predictor.predict_batch(
        multimask_output=False,
        point_coords_batch=input_point if use_prompt else None,
        point_labels_batch=input_label if use_prompt else None
    )
    num_objects = 1
    metric_dict = estimate(masks, gt_mask, num_objects, metric_dict)

for (metric_name, _), score_list in metric_dict.items():
    print(f"Average {metric_name}:", np.mean(score_list))