import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_unlabeled_dataset import VOCUnlabeledDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Paths
root_unlabeled = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_test"
checkpoint = "deeplab_resnet50_voc_full.pth"
output_dir = "pseudo_labels"
os.makedirs(output_dir, exist_ok=True)

# Load teacher model (21 classes)
model = deeplabv3_resnet50(weights=None, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)
model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
model.eval()

# Unlabeled dataset
dataset = VOCUnlabeledDataset(root_unlabeled, target_size=(256, 256))
loader = DataLoader(dataset, batch_size=1)

print(f"Generating pseudo-labels for {len(dataset)} images...")

# Generate pseudo-labels
for i, image in enumerate(loader):
    with torch.no_grad():
        output = model(image)["out"]
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Save pseudo-mask
    mask = Image.fromarray(pred.astype(np.uint8))
    mask.save(os.path.join(output_dir, f"pseudo_{i:05d}.png"))

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(dataset)} images")

print(f"✅ Pseudo-labels saved in {output_dir}")
