import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_dataset import VOCDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# --------------------------
# Configuration
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "deeplab_student_full.pth"
root_val = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"
save_dir = "results_student_full_eval"
os.makedirs(save_dir, exist_ok=True)

num_classes = 21
target_size = (256, 256)

# VOC 2012 class names (21 including background)
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]

# --------------------------
# Load validation dataset
# --------------------------
print("📂 Loading validation dataset...")
val_dataset = VOCDataset(root_val, split="val", target_size=target_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"✅ Validation samples: {len(val_dataset)}")

# --------------------------
# Load trained model
# --------------------------
print("🧠 Loading trained student model...")
model = deeplabv3_resnet50(weights=None, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval().to(device)

# --------------------------
# Metric helpers
# --------------------------
def pixel_accuracy(pred, label):
    valid = (label >= 0) & (label < num_classes)
    correct = (pred[valid] == label[valid]).sum()
    total = valid.sum()
    return correct.float() / total.float()

def intersection_over_union_per_class(pred, label, num_classes):
    ious = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = label == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious[cls] = np.nan
        else:
            ious[cls] = (intersection / union).item()
    return ious

# --------------------------
# Evaluation
# --------------------------
print("🚀 Running evaluation and saving masks...")
acc_list, iou_sum = [], np.zeros(num_classes)
count_valid = 0

with torch.no_grad():
    for idx, (images, masks) in enumerate(val_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1)

        # Metrics
        acc = pixel_accuracy(preds, masks)
        ious = intersection_over_union_per_class(preds.cpu().squeeze(0), masks.cpu().squeeze(0), num_classes)

        acc_list.append(acc.item())
        iou_sum += np.nan_to_num(ious)
        count_valid += (~np.isnan(ious)).astype(int)

        # Save predicted mask
        pred_np = preds.cpu().squeeze(0).numpy().astype(np.uint8)
        mask_img = Image.fromarray(pred_np)
        mask_img.save(os.path.join(save_dir, f"pred_{idx:04d}.png"))

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(val_loader)} images...")

# --------------------------
# Results
# --------------------------
mean_acc = np.nanmean(acc_list)
class_iou = iou_sum / np.maximum(count_valid, 1)
mean_miou = np.nanmean(class_iou)

print("\n🎯 Final Evaluation Results:")
print(f"✅ Pixel Accuracy: {mean_acc * 100:.2f}%")
print(f"✅ Mean IoU: {mean_miou * 100:.2f}%")
print("\n📊 Per-Class IoU:")
for cls_name, iou in zip(CLASS_NAMES, class_iou):
    print(f"{cls_name:>15s}: {iou * 100:.2f}%")

# --------------------------
# Save report
# --------------------------
report_path = os.path.join(save_dir, "metrics_detailed.txt")
with open(report_path, "w") as f:
    f.write(f"Pixel Accuracy: {mean_acc * 100:.2f}%\n")
    f.write(f"Mean IoU: {mean_miou * 100:.2f}%\n\n")
    f.write("Per-Class IoU:\n")
    for cls_name, iou in zip(CLASS_NAMES, class_iou):
        f.write(f"{cls_name:>15s}: {iou * 100:.2f}%\n")

print(f"\n📁 All predicted masks and metrics saved to: {save_dir}")
