import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VOCUnlabeledDataset(Dataset):
    def __init__(self, root, target_size=(256, 256)):
        """
        Unlabeled VOC dataset (only images, no masks).
        Args:
            root: path to VOC2012_test folder
            target_size: resize images for training consistency
        """
        self.root = root
        self.target_size = target_size
        image_dir = os.path.join(root, "JPEGImages")

        # Collect all jpg images
        self.images = [os.path.join(image_dir, f)
                       for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        # Resize to fixed size
        image = image.resize(self.target_size, Image.BILINEAR)

        # Normalize to [0,1] and convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # [C,H,W]

        return image
