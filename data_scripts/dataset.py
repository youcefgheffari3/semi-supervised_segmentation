import torch
from torch.utils.data import Dataset
import numpy as np

class RandomSegmentationDataset(Dataset):
    def __init__(self, num_samples=100, image_size=128, num_classes=21):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random RGB image
        image = np.random.randint(0, 256, (3, self.image_size, self.image_size), dtype=np.uint8)
        # Random segmentation mask (values = class IDs)
        mask = np.random.randint(0, self.num_classes, (self.image_size, self.image_size), dtype=np.uint8)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask
