import torch
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_dataset import VOCDataset

# Path to dataset
root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"

# Validation dataset
dataset = VOCDataset(root, split="val", target_size=(256, 256))

# ✅ Build DeepLabV3 model (same as training, 21 classes)
model = deeplabv3_resnet50(weights=None)  # no pretrained weights here
model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)

# Load trained checkpoint
model.load_state_dict(torch.load("deeplab_resnet50_voc_frozen.pth"))
model.eval()

# Pick one sample from validation
image, mask = dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))["out"]
    pred = torch.argmax(output, dim=1).squeeze(0)

# Show results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image.permute(1, 2, 0).numpy())
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(mask.numpy(), cmap="tab20")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred.numpy(), cmap="tab20")
plt.title("Prediction (DeepLabV3 Frozen)")

plt.show()
