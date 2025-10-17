import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from voc_dataset import VOCDataset

# Path to dataset
root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"

# Dataset (resize + subset for CPU speed)
full_dataset = VOCDataset(root, split="train", target_size=(256, 256))
train_dataset = Subset(full_dataset, range(200))  # small subset
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ✅ Load DeepLabV3 with ResNet50 backbone (pretrained weights)
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights)

# Replace classifier head with 21 classes (VOC)
model.classifier[4] = nn.Conv2d(256, 21, kernel_size=1)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(20):  # just 1 epoch for testing
    running_loss = 0.0
    for images, masks in train_loader:
        outputs = model(images)["out"]  # DeepLab returns dict with 'out'
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/1], Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "deeplab_resnet50_voc.pth")
print("Model saved as deeplab_resnet50_voc.pth")
