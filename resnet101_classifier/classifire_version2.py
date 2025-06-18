import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from models.classifier.classifier_dataset import ClassifierDataset


# Classifier model using ResNet101
class ResNet101Classifier(nn.Module):
    def __init__(self, num_channels=84, num_classes=4):
        super(ResNet101Classifier, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # Modify the first convolutional layer to accept 84 channels
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer to output 4 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Training loop with TensorBoard logging
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.writer = SummaryWriter('/Users/ramtahor/PycharmProjects/thesis/resnet101_classifier/runs')

    def train(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Log statistics
                running_loss += loss.item()
                if i % 10 == 9:  # log every 10 batches
                    print(
                        f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {running_loss / 10:.4f}')
                    self.writer.add_scalar('training loss', running_loss / 10, epoch * len(self.train_loader) + i)
                    running_loss = 0.0
            # Evaluate on validation data after each epoch
            val_loss, val_accuracy = self.evaluate()
            print(f'Epoch[{epoch + 1}/{self.num_epochs}], Validation loss: {val_loss / 10: .4f}, Validation accuracy: {val_accuracy: .4f}')
            self.writer.add_scalar('Validation loss', val_loss / 10, epoch)
            self.writer.add_scalar('Validation accuracy', val_accuracy, epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return average_loss, accuracy

    def close(self):
        self.writer.close()


# Example usage
if __name__ == "__main__":
    # Load your custom dataset here
    # This assumes that your custom dataset inherits from torch.utils.data.Dataset
    # Replace 'YourCustomDataset' with your dataset class name
    # train_dataset = YourCustomDataset()  # Ensure your dataset class is defined with appropriate transformations and loading methods
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    dataset = ClassifierDataset('/home/ARO.local/tahor/PycharmProjects/data/pair_data', transform=False, state='before')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Model, loss function, optimizer
    model = ResNet101Classifier(num_channels=84, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)
    trainer.train()
    trainer.close()
