import torch
import torch.nn as nn
import torch.optim as optim
from sympy.stats.sampling.sample_numpy import numpy
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from resnet101_classifier.dataset import ClassifierDataset
from torch.utils.tensorboard import SummaryWriter
import clearml_agent
from tqdm import tqdm
import config

# Set CUDA_LAUNCH_BLOCKING to 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"CUDA_LAUNCH_BLOCKING is set to: {os.environ['CUDA_LAUNCH_BLOCKING']}")

cudnn.benchmark = True

def imshow(inp, fname: str, inp_channels=None, title=None):
    """Display image for Tensor."""
    if inp_channels is None:
        inp_channels = [44, 55, 60]
    inp = inp.numpy().transpose((1, 2, 0))[:, :, inp_channels]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imsave(f'{fname}.png', inp)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def calculate_accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)

def calculate_precision(preds, labels):
    true_positives = ((preds == labels) & (labels == 1)).sum().item()
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, train_writer, val_writer, num_epochs=25):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        best_acc = 0.0
        print('model saved')

        for epoch in tqdm(range(num_epochs), desc='Train Model'):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(config.DEVICE)
                    labels = labels.to(config.DEVICE)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_precision = calculate_precision(np.array(all_preds), np.array(all_labels))

                if phase == 'train':
                    scheduler.step()
                    train_writer.add_scalar('Loss/train', epoch_loss, epoch)
                    train_writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                    train_writer.add_scalar('Precision/train', epoch_precision, epoch)
                else:
                    val_writer.add_scalar('Loss/val', epoch_loss, epoch)
                    val_writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                    val_writer.add_scalar('Precision/val', epoch_precision, epoch)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    save_checkpoint(model, optimizer)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
        val_writer.close()
        train_writer.close()
        return model

def visualize_model(model, dataloaders, train_writer, val_writer, num_images=1):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                imshow(inputs.cpu().data[j], 'inputs')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            print('inputs size:', (inputs.size()))

    model.train(mode=was_training)

def main():
    dataset = ClassifierDataset('/home/ARO.local/tahor/PycharmProjects/data/pair_data', channels=840, c_step=10, transform=False, state='before')

    print('line 204')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    data_dict = {'train': train_dataset, 'val': test_dataset}
    dataset_sizes = {x: len(data_dict[x]) for x in ['train', 'val']}

    print('line 221')

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True),
                   'val': torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)}

    valwriter = SummaryWriter('run_experiments/remote/classifier')
    trainwriter = SummaryWriter('run_experiments/remote/classifier')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    model_ft.conv1 = nn.Conv2d(84, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, trainwriter, valwriter, num_epochs=100)

    visualize_model(model_ft, dataloaders, trainwriter, valwriter)
    print('vis model')

if __name__ == '__main__':
    main()