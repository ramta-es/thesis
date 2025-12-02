import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from resnet101_classifier.dataset import ClassifierDataset

class ResNet101Classifier(nn.Module):
    def __init__(self, num_channels=84, num_classes=4):
        super().__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=10, log_dir='runs', device='cpu', biased_class=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir)
        self.device = device
        self.biased_class = biased_class

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if (i + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    avg_acc = correct / total if total > 0 else 0
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                    self.writer.add_scalar('training loss', avg_loss, epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('training accuracy', avg_acc, epoch * len(self.train_loader) + i)
                    running_loss = 0.0
                    correct = 0
                    total = 0
            val_loss, val_acc, biased_acc = self.evaluate()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}, Biased accuracy: {biased_acc:.4f}')
            self.writer.add_scalar('Validation loss', val_loss, epoch)
            self.writer.add_scalar('Validation accuracy', val_acc, epoch)
            if self.biased_class is not None:
                self.writer.add_scalar(f'Biased accuracy/class_{self.biased_class}', biased_acc, epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        biased_correct = 0
        biased_total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if self.biased_class is not None:
                    mask = (labels == self.biased_class)
                    if mask.any():
                        biased_correct += (predicted[mask] == labels[mask]).sum().item()
                        biased_total += mask.sum().item()
        average_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        biased_accuracy = biased_correct / biased_total if biased_total > 0 else 0
        return average_loss, accuracy, biased_accuracy

    def close(self):
        self.writer.close()

def main(data_dir, num_classes, num_epochs, batch_size, log_dir, device, biased_class=None):
    dataset = ClassifierDataset(data_dir, channels=84, c_step=1, transform=False, state='before')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = ResNet101Classifier(num_channels=84, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs, log_dir, device, biased_class)
    trainer.train()
    trainer.close()