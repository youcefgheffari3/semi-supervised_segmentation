import torch
import matplotlib.pyplot as plt
from model import SimpleSegNet
from dataset import RandomSegmentationDataset

# Load model
num_classes = 21
model = SimpleSegNet(num_classes=num_classes)
model.eval()

# Fake dataset
dataset = RandomSegmentationDataset(num_samples=1, image_size=64, num_classes=num_classes)
image, mask = dataset[0]

# Add batch dimension
image_batch = image.unsqueeze(0)

# Forward pass
with torch.no_grad():
    output = model(image_batch)
    pred = torch.argmax(output, dim=1).squeeze(0)

# Show input + mask + prediction
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image.permute(1, 2, 0).numpy())

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(mask.numpy(), cmap="tab20")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred.numpy(), cmap="tab20")

plt.show()
