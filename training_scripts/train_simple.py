import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import SimpleSegNet
from voc_dataset import VOCDataset

# Path to your dataset
root = r"C:/Users/Gheffari Youcef/Documents/Project/datasets/VOC2012_train_val"  # adjust if needed

# Dataset & loader (⚡ resize smaller and use only 100 samples for speed)
full_dataset = VOCDataset(root, split="train", target_size=(128, 128))
train_dataset = Subset(full_dataset, range(100))  # use only first 100 images
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Model, loss, optimizer
num_classes = 21
model = SimpleSegNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 = ignore label in VOC
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (⚡ just 1 epoch for speed)
for epoch in range(1):
    running_loss = 0.0
    for images, masks in train_loader:
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/1], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "simple_segnet_voc_fast.pth")
print("Model saved as simple_segnet_voc_fast.pth")
