import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import transforms
from voc_dataset import VOCDataset

# -----------------------------
# Config
# -----------------------------
root = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"
num_classes = 21
batch_size = 4   # increase if you have GPU
num_epochs = 50
learning_rate = 1e-4

# -----------------------------
# Data Augmentations
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(256, scale=(0.5, 2.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])

# -----------------------------
# Dataset & DataLoader
# -----------------------------
train_dataset = VOCDataset(root, split="train", target_size=(256, 256))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Model Setup
# -----------------------------
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights, aux_loss=True)

# Replace classifier head with VOC-specific head
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

# Unfreeze entire model (fine-tuning backbone + head)
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler: reduce LR every 15 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        # Apply augmentations on the fly
        for i in range(len(images)):
            pil_img = transforms.ToPILImage()(images[i])
            pil_img = train_transforms(pil_img)
            images[i] = transforms.ToTensor()(pil_img)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        ckpt_path = f"deeplab_resnet50_voc_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

# Save final model
torch.save(model.state_dict(), "deeplab_resnet50_voc_full.pth")
print("✅ Training complete. Model saved as deeplab_resnet50_voc_full.pth")
