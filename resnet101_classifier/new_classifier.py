import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import copy
import os
from dataset import ClassifierDataset


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
    import copy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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
                    # loss = criterion(outputs, labels)

                    # labels_onehot = F.one_hot(labels, num_classes=4).float()
                    # loss = criterion(outputs, labels_onehot)
                    loss = criterion(outputs, labels.float())

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

            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)
            writer.flush()

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Save and load best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_model.pth')
                model.load_state_dict(best_model_wts)  # Load best weights immediately

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history


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


def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('history_plot_2_classes_crossentropy_loss.png')  # Save the plot automatically
    plt.show()



def main():
    num_classes = 2
    batch_size = 20
    num_epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create synthetic dataset
    # train_dataset = SyntheticDataset(num_samples=20, image_size=(84, 550, 550), num_classes=num_classes)
    # val_dataset = SyntheticDataset(num_samples=5, image_size=(84, 550, 550), num_classes=num_classes)
    dataset = ClassifierDataset('/home/ARO.local/tahor/PycharmProjects/data/pair_data', channels=840, c_step=10,
                                transform=False, state='before')


    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    data_dict = {'train': train_dataset, 'val': test_dataset}
    dataset_sizes = {x: len(data_dict[x]) for x in ['train', 'val']}



    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True),
                   'val': torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)}

    # Initialize model, criterion, optimizer, and scheduler
    model = initialize_model(num_classes)
    model = model.to(device)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # TensorBoard writer
    writer = SummaryWriter('/home/ARO.local/tahor/PycharmProjects/thesis/resnet101_classifier/runs')

    # Train the model and get history
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device, writer)

    # Add model graph to TensorBoard
    sample_input = torch.randn(1, 84, 550, 550).to(device)
    writer.add_graph(model, sample_input)

    # Plot training and validation loss/accuracy
    plot_history(history)


    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()