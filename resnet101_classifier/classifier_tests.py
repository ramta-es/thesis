import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from resnet101_classifier.classifier_V3 import ResNet101Classifier, Trainer

class SyntheticClassifierDataset(Dataset):
    def __init__(self, num_samples=4, num_channels=84, height=64, width=64, num_classes=4):
        self.images = np.random.rand(num_samples, num_channels, height, width).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])

class TestResNet101Classifier(unittest.TestCase):
    def setUp(self):
        self.num_classes = 4
        self.num_channels = 84
        self.dataset = SyntheticClassifierDataset(num_samples=4, num_channels=self.num_channels, num_classes=self.num_classes)
        self.loader = DataLoader(self.dataset, batch_size=2)  # batch_size >= 2
        self.model = ResNet101Classifier(num_channels=self.num_channels, num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def test_forward_pass(self):
        for images, labels in self.loader:
            outputs = self.model(images)
            self.assertEqual(outputs.shape[0], images.shape[0])
            self.assertEqual(outputs.shape[1], self.num_classes)

    def test_training_step(self):
        trainer = Trainer(self.model, self.loader, self.loader, self.criterion, self.optimizer, num_epochs=1, device='cpu')
        try:
            trainer.train()
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")
        finally:
            trainer.close()

if __name__ == '__main__':
    unittest.main()
