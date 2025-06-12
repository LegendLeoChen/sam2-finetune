import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import sys

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.loss_fn import total_loss
from training.dataset import LabPicsDataset, SemSegDataset
from tqdm import tqdm

os.chdir(sys.path[0])

# SAM2前向传播
def forward_net(predictor, image_list, use_prompt=False, input_point=None, input_label=None):
    predictor.set_image_batch(image_list)  # 应用图像编码器
    if use_prompt:
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)      # 有提示
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
    else:
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=None, boxes=None, masks=None)       # 无提示
    
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features
    )
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # 将掩码上采样到原始图像分辨率
    return prd_masks, prd_scores, torch.sigmoid(prd_masks[:, 0])

# 训练一代
def train_epoch(predictor, train_dataloader, epoch, optimizer, scaler, scheduler, writer, use_prompt=False, accumulation_steps=4):
    step = len(train_dataloader) * epoch
    progress_bar = tqdm(train_dataloader, colour='green')
    for batch_idx, (image, mask, input_point, input_label) in enumerate(progress_bar):
        gt_mask = mask.float().to(device)
        step += 1
        with autocast('cuda', torch.bfloat16):  # 混合精度训练
            prd_masks, prd_scores, prd_mask = forward_net(predictor, image, use_prompt, input_point if use_prompt else None, input_label if use_prompt else None)
            loss, losses = total_loss(inputs=prd_masks, targets=gt_mask, pred_ious=prd_scores, num_objects=image.shape[0],
                                    dice_weight=0.4, focal_weight=12.0, diou_weight=5.0, iou_weight=2.0, giou_weight=0.2)

        loss = loss / accumulation_steps

        # 反向传播/梯度累积
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.3f}',
                'Dice Loss': f'{losses["dice"].item():.2f}',
                'Focal Loss': f'{losses["focal"].item():.2f}',
                'dIOU Loss': f'{losses["diou"].item():.2f}',
                'IOU Loss': f'{losses["iou"].item():.2f}'
            })

        writer.add_scalar('train/total loss', loss.item() * accumulation_steps, step)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        for key, value in losses.items():
            writer.add_scalar(f'train/{key}loss', value.item(), step)
            
    scheduler.step()            # 每代结束学习率更新

# 验证
def validate(predictor, val_dataloader, epoch, writer, use_prompt=False):
    predictor.model.sam_mask_decoder.train(False)
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (image, mask, input_point, input_label) in enumerate(tqdm(val_dataloader, colour='blue')):    
            gt_mask = mask.float().to(device)
            prd_masks, prd_scores, prd_mask = forward_net(predictor, image, use_prompt, input_point if use_prompt else None, input_label if use_prompt else None)
            val_loss, val_losses = total_loss(inputs=prd_masks, targets=gt_mask, pred_ious=prd_scores, num_objects=image.shape[0],
                                    dice_weight=0.4, focal_weight=12.0, diou_weight=5.0, iou_weight=2.0, giou_weight=0.2)
            total_val_loss += val_loss.item()
    total_val_loss /= len(val_dataloader)
    predictor.model.sam_mask_decoder.train(True)
    print(f"Epoch {epoch + 1}, Val Loss: {total_val_loss:.3f}")
    writer.add_scalar('val/total_loss', total_val_loss, epoch + 1)


if __name__ == "__main__":
    # 配置参数
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 32                     # 批次大小
    model_scale_dict = {"b+": "base_plus", "l": "large", "s": "small", "t": "tiny"}
    model_scale = "l"                   # 模型大小
    learning_rate = 2e-4                # 学习率
    weight_decay = 4e-5                 # 权重衰减
    T_max = 40                         # 余弦调度器半周期
    accumulation_steps = 4              # 梯度积累
    use_prompt = True                  # 是否使用提示
    epoches = 50                       # 迭代数
    use_pretrained_model = True         # 是否用预训练权重
    pretrained_pt = (16, 50)           # exp{0}文件夹下的{1}epoch权重
    eval_interval = 1                  # 验证间隔
    save_interval = 10                  # 保存权重间隔

    # 定义颜色增强
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(
            brightness=0.3,     # 亮度调整范围
            contrast=0.3,       # 对比度调整范围
            saturation=0.3,     # 饱和度调整范围
            hue=0.2             # 色调调整范围
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 验证集只做归一化
    ])

    # 数据集和数据加载器
    # train_dataset = LabPicsDataset("../assets/LabPicsV1/Simple/Train", transform=train_transform)
    train_dataset = SemSegDataset("../../datasets", dataset_type="training", transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_dataset = LabPicsDataset("../assets/LabPicsV1/Simple/Test", transform=val_transform)
    val_dataset = SemSegDataset("../../datasets", dataset_type="validation", transform=transforms.ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 加载模型
    sam2_checkpoint = f"../checkpoints/sam2_hiera_{model_scale_dict[model_scale]}.pt"
    model_cfg = f"../sam2_configs/sam2_hiera_{model_scale}.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    if use_pretrained_model:
        predictor.model.load_state_dict(torch.load(f"./output/exp{pretrained_pt[0]}/model_epoch_{pretrained_pt[1]}.pt"))

    # 设置训练参数
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(False)
    predictor.model.image_encoder.train(False)
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=5e-8)
    scaler = GradScaler('cuda')
    writer = SummaryWriter(log_dir="output")

    print("train on")
    # 训练循环
    for epoch in range(epoches):
        # 训练环节
        train_epoch(predictor, train_dataloader, epoch, optimizer, scaler, scheduler, writer, use_prompt, accumulation_steps)
        # 验证环节
        if (epoch + 1) % eval_interval == 0:
            validate(predictor, val_dataloader, epoch, writer, use_prompt)

        if (epoch + 1) % save_interval == 0:
            torch.save(predictor.model.state_dict(), f"./output/model_epoch_{epoch + 1}.pt")

    writer.close()