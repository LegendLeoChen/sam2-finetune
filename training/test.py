import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.metric import *
from training.dataset import LabPicsDataset

os.chdir(sys.path[0])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
model_scale_dict = {"b+": "base_plus", "l": "large", "s": "small", "t": "tiny"}
model_scale = "t"

# 数据集和数据加载器
dataset = LabPicsDataset("../assets/LabPicsV1/Simple/Test", transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 加载模型
sam2_checkpoint = f"../checkpoints/sam2_hiera_{model_scale_dict[model_scale]}.pt"
model_cfg = f"../sam2_configs/sam2_hiera_{model_scale}.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(f"./output/exp{15}/model_epoch_{250}.pt"))

use_prompt = False

iou_scores = []
dice_scores = []
pixel_acc_scores = []
boundary_f1_scores = []

for batch_idx, (image, mask, input_point, input_label) in enumerate(dataloader):    
    gt_mask = mask.float().unsqueeze(1)
    gt_mask = [gt_mask[i].numpy() for i in range(gt_mask.shape[0])]
    predictor.set_image_batch(image)
    masks, scores, logits = predictor.predict_batch(
        multimask_output=False,
    )

    num_objects = 1

    for output, target in zip(masks, gt_mask):
        iou_scores.append(iou(output, target, num_objects))
        dice_scores.append(dice(output, target, num_objects))
        pixel_acc_scores.append(pixel_accuracy(output, target, num_objects))
        boundary_f1_scores.append(boundary_f1_score(output, target, num_objects))

print("Average IoU:", np.mean(iou_scores))
print("Average Dice:", np.mean(dice_scores))
print("Average Pixel Accuracy:", np.mean(pixel_acc_scores))
print("Average Boundary F1 Score:", np.mean(boundary_f1_scores))
        