import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_dataset import VOCDataset

# VOC 2012 class names (21 including background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

def compute_confusion_matrix(pred, target, num_classes, ignore_index=255):
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(target.view(-1), pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix

def compute_ious(conf_matrix):
    ious = []
    for i in range(conf_matrix.shape[0]):
        intersection = conf_matrix[i, i].item()
        union = conf_matrix[i, :].sum().item() + conf_matrix[:, i].sum().item() - intersection
        if union == 0:
            ious.append(float("nan"))  # skip class not present
        else:
            ious.append(intersection / union)
    return ious

# -----------------------------
# Load dataset
# -----------------------------
root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"
dataset = VOCDataset(root, split="val", target_size=(256, 256))

# -----------------------------
# Load trained model
# -----------------------------
model = deeplabv3_resnet50(weights=None, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)
state_dict = torch.load("deeplab_resnet50_voc.pth")  # or frozen version
model.load_state_dict(state_dict, strict=False)
model.eval()

# -----------------------------
# Evaluation loop
# -----------------------------
num_classes = 21
conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

print("Evaluating on", len(dataset), "validation images...")
with torch.no_grad():
    for i in range(len(dataset)):
        image, mask = dataset[i]
        output = model(image.unsqueeze(0))["out"]
        pred = torch.argmax(output, dim=1).squeeze(0)
        conf_matrix += compute_confusion_matrix(pred, mask, num_classes)

# -----------------------------
# Compute IoUs
# -----------------------------
ious = compute_ious(conf_matrix)
mean_iou = np.nanmean(ious)

print("\n✅ Mean IoU on VOC val set: {:.4f}".format(mean_iou))
print("\nPer-class IoU:")
for cls_name, iou in zip(VOC_CLASSES, ious):
    if not np.isnan(iou):
        print(f"{cls_name:>12}: {iou:.4f}")
    else:
        print(f"{cls_name:>12}: n/a (not present in GT)")

# -----------------------------
# Save results to CSV
# -----------------------------
csv_path = "voc_iou_results.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "IoU"])
    for cls_name, iou in zip(VOC_CLASSES, ious):
        if not np.isnan(iou):
            writer.writerow([cls_name, f"{iou:.4f}"])
        else:
            writer.writerow([cls_name, "n/a"])
    writer.writerow([])
    writer.writerow(["Mean IoU", f"{mean_iou:.4f}"])

print(f"\n📂 Results saved to {csv_path}")

# -----------------------------
# Plot per-class IoU bar chart
# -----------------------------
valid_classes = [c for c, iou in zip(VOC_CLASSES, ious) if not np.isnan(iou)]
valid_ious = [iou for iou in ious if not np.isnan(iou)]

plt.figure(figsize=(14, 6))
bars = plt.bar(valid_classes, valid_ious, color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("IoU")
plt.title("Per-class IoU on VOC 2012 Validation Set")

# Annotate bars with values
for bar, iou in zip(bars, valid_ious):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{iou:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("voc_iou_results.png")
plt.show()

print("\n📊 Per-class IoU bar chart saved as voc_iou_results.png")
