from voc_dataset import VOCDataset
import matplotlib.pyplot as plt

root = r"C:/Users/Gheffari Youcef/Documents/Project/datasets/VOC2012_train_val"  # adjust this path

dataset = VOCDataset(root, split="train")
print("Dataset size:", len(dataset))

image, mask = dataset[0]
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

# Show example
plt.subplot(1,2,1)
plt.imshow(image.permute(1,2,0).numpy())
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask.numpy(), cmap="tab20")
plt.title("Segmentation Mask")
plt.show()
