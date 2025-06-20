import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import sys
import cv2

os.chdir(sys.path[0])
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2_configs/sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
# predictor.model.load_state_dict(torch.load("./output/exp13/model_epoch_100.pt"))

# 预测单图
# image = Image.open('../notebooks/images/truck.jpg')
# image = Image.open('../assets/LabPicsV1/Simple/Test/Image/Sagi_Fire that change color in response to a magnetic field-screenshot (1).jpg')
image = Image.open('../../datasets/ade20k/images/validation/ADE_val_00000128.jpg')
image = np.asarray(image.convert("RGB"))
predictor.set_image(image)

input_point = None
input_label = None
input_point = np.array([[200, 250]])
input_label = np.array([1])

a = time.time()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
print(time.time() - a)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

# 预测多图
# image1 = Image.open('../notebooks/images/truck.jpg')
# image1 = np.asarray(image1.convert("RGB"))
# image1_boxes = np.array([
#     [75, 275, 1725, 850],
#     [425, 600, 700, 875],
#     [1375, 550, 1650, 800],
#     [1240, 675, 1400, 750],
# ])

# image2 = Image.open('../notebooks/images/groceries.jpg')
# image2 = np.asarray(image2.convert("RGB"))
# image2_boxes = np.array([
#     [450, 170, 520, 350],
#     [350, 190, 450, 350],
#     [500, 170, 580, 350],
#     [580, 170, 640, 350],
# ])

# img_batch = [image1, image2]
# boxes_batch = [image1_boxes, image2_boxes]

# predictor.set_image_batch(img_batch)

# masks_batch, scores_batch, _ = predictor.predict_batch(
#     None,
#     None, 
#     box_batch=boxes_batch, 
#     multimask_output=False
# )

# for image, boxes, masks in zip(img_batch, boxes_batch, masks_batch):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     for mask in masks:
#         show_mask(mask.squeeze(0), plt.gca(), random_color=True)
#     for box in boxes:
#         show_box(box, plt.gca())

# plt.show()
