import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Synthetic dataset class
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=2, image_size=(84, 550, 550), num_classes=4):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = [torch.rand(*image_size) for _ in range(num_samples)]
        self.labels = [np.random.randint(0, num_classes) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Test the synthetic dataset
if __name__ == "__main__":
    # Create synthetic dataset
    synthetic_dataset = SyntheticDataset()
    dataloader = DataLoader(synthetic_dataset, batch_size=1, shuffle=True)

    # Verify the dataset
    for images, labels in dataloader:
        print(f"Image shape: {images.shape}, Label: {labels}")