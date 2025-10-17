import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VOCPseudoDataset(Dataset):
    def __init__(self, root, mask_dir, target_size=(256, 256)):
        """
        Pseudo-labeled VOC dataset.
        Args:
            root: VOC2012_test folder (with JPEGImages)
            mask_dir: folder where pseudo masks are stored
            target_size: resize size for images and masks
        """
        self.root = root
        self.mask_dir = mask_dir
        self.target_size = target_size

        image_dir = os.path.join(root, "JPEGImages")
        self.images = sorted([os.path.join(image_dir, f)
                              for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.masks = sorted([os.path.join(mask_dir, f)
                             for f in os.listdir(mask_dir) if f.endswith(".png")])

        assert len(self.images) == len(self.masks), "Mismatch between images and masks!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert("RGB")
        image = image.resize(self.target_size, Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        # Load pseudo mask
        mask = Image.open(self.masks[idx])
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask = np.array(mask).astype(np.int64)
        mask = torch.tensor(mask)

        return image, mask
