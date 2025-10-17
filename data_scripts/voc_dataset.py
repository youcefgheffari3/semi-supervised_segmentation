import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_size=(256, 256)):
        """
        root: path to VOC2012_train_val
        split: "train" or "val"
        target_size: resize all images/masks to this size
        """
        self.root = root
        self.transform = transform
        self.target_size = target_size

        image_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "SegmentationClass")

        split_file = os.path.join(root, "ImageSets", "Segmentation", split + ".txt")
        with open(split_file) as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        # ✅ Resize both image & mask
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)  # NEAREST keeps class IDs intact

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.int64)

        image = torch.tensor(image).permute(2, 0, 1)  # [C,H,W]
        mask = torch.tensor(mask)                     # [H,W]

        return image, mask
