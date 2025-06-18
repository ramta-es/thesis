import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import os
from try_class_new import SyntheticDataset
from PIL import Image
# Define the dataset (replace with your custom dataset)
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx])
        label = int(self.data[idx].split('_')[0])  # Assuming label is in the filename
        image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the model
def initialize_model(num_classes):
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    # Modify the first convolutional layer to accept 84 input channels
    model.conv1 = nn.Conv2d(84, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training function
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log loss and accuracy to TensorBoard
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)

            # Flush the writer to ensure logs are written
            writer.flush()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = running_corrects.double() / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Main function
# def main():
#     data_dir = '/path/to/data'  # Replace with your dataset path
#     num_classes = 4
#     batch_size = 16
#     num_epochs = 25
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Data transformations
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize((550, 550)),  # Resize to 550x550
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485] * 84, [0.229] * 84)  # Normalize for 84 channels
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((550, 550)),  # Resize to 550x550
#             transforms.ToTensor(),
#             transforms.Normalize([0.485] * 84, [0.229] * 84)  # Normalize for 84 channels
#         ]),
#     }
#     # Datasets and dataloaders
#     image_datasets = {x: CustomDataset(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
#     dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#
#     # Initialize model, criterion, optimizer, and scheduler
#     model = initialize_model(num_classes)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
#     # TensorBoard writer
#     writer = SummaryWriter('runs/resnet101_classifier')
#
#     # Add model graph to TensorBoard
#     sample_input = torch.randn(1, 3, 224, 224).to(device)
#     writer.add_graph(model, sample_input)
#
#     # Train the model
#     model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer)
#
#     # Evaluate the model
#     test_accuracy = evaluate_model(model, dataloaders['val'], device)
#
#     # Close the TensorBoard writer
#     writer.close()



def main():
    num_classes = 4
    batch_size = 4
    num_epochs = 25
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create synthetic dataset
    train_dataset = SyntheticDataset(num_samples=20, image_size=(84, 550, 550), num_classes=num_classes)
    val_dataset = SyntheticDataset(num_samples=5, image_size=(84, 550, 550), num_classes=num_classes)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Initialize model, criterion, optimizer, and scheduler
    model = initialize_model(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # TensorBoard writer
    writer = SummaryWriter('/Users/ramtahor/PycharmProjects/thesis/resnet101_classifier/runs')

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer)

    # Add model graph to TensorBoard
    sample_input = torch.randn(1, 84, 550, 550).to(device)
    writer.add_graph(model, sample_input)


    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()