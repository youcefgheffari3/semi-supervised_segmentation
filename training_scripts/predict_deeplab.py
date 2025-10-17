import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_dataset import VOCDataset

# Path to dataset
root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"

# VOC 2012 class names (21 including background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

num_classes = len(VOC_CLASSES)

# Validation dataset
dataset = VOCDataset(root, split="val", target_size=(256, 256))

# ✅ Build DeepLabV3 model with aux head (same as training)
model = deeplabv3_resnet50(weights=None, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

# ✅ Load trained checkpoint
state_dict = torch.load("deeplab_resnet50_voc_full.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Function to compute IoU per class
def compute_iou_per_class(pred, target, num_classes=21, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float("nan"))  # no samples for this class
    return ious

# ✅ Compute IoUs across validation set
total_ious = [[] for _ in range(num_classes)]

for i in range(len(dataset)):  # ⚠ may take long on CPU
    image, mask = dataset[i]
    with torch.no_grad():
        output = model(image.unsqueeze(0))["out"]
        pred = torch.argmax(output, dim=1).squeeze(0)

    ious = compute_iou_per_class(pred, mask, num_classes=num_classes)
    for cls, iou in enumerate(ious):
        if not np.isnan(iou):
            total_ious[cls].append(iou)

# Average per-class IoUs
mean_ious = [np.mean(cls_ious) if len(cls_ious) > 0 else float("nan") for cls_ious in total_ious]
mIoU = np.nanmean(mean_ious)

# ✅ Save results to CSV
csv_file = "voc_iou_results.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "IoU"])
    for cls, score in enumerate(mean_ious):
        writer.writerow([VOC_CLASSES[cls], f"{score:.4f}" if not np.isnan(score) else "N/A"])
    writer.writerow([])
    writer.writerow(["Mean IoU", f"{mIoU:.4f}"])

print(f"📊 Per-class IoU scores saved to {csv_file}")
print(f"✅ Mean IoU (mIoU): {mIoU:.4f}")

# ✅ Bar chart of per-class IoUs
plt.figure(figsize=(12, 6))
scores_to_plot = [score if not np.isnan(score) else 0 for score in mean_ious]
plt.bar(range(num_classes), scores_to_plot, tick_label=VOC_CLASSES)
plt.xticks(rotation=45, ha="right")
plt.ylabel("IoU")
plt.title(f"Per-class IoU (mIoU={mIoU:.4f})")
plt.tight_layout()
plt.savefig("voc_iou_results.png")  # save figure as PNG
plt.show()

# ✅ Visualize one example
image, mask = dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))["out"]
    pred = torch.argmax(output, dim=1).squeeze(0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image.permute(1, 2, 0).numpy())
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(mask.numpy(), cmap="tab20")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred.numpy(), cmap="tab20")
plt.title("Prediction (DeepLabV3)")

plt.show()
