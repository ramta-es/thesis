"""
Training script for a modified ResNet101 classifier with multi-channel input.
Adds collection of per-epoch metrics (loss & accuracy) and saves a plot
`training_curves.png` at the end of training.
"""

import os
import time
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from PIL import Image  # kept for potential future inference extension
import matplotlib
matplotlib.use('Agg')  # Add this at the very top, before importing pyplot
import matplotlib.pyplot as plt
from resnet101_classifier.dataset import ClassifierDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CUDA setup
cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def imshow(tensor, fname: str, channel_indices=None):
    """
    Save a quick-look composite image selecting given channels.

    Args:
        tensor (Tensor): Image tensor (C,H,W).
        fname (str): Output filename prefix (PNG added automatically).
        channel_indices (list[int] | None): Channels to visualize (default selects pseudo-RGB).
    """
    if channel_indices is None:
        channel_indices = [44, 55, 60]
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr[:, :, channel_indices]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = std * arr + mean
    arr = np.clip(arr, 0, 1)
    try:
        plt.imsave(f'{fname}.png', arr)
    except Exception:
        pass  # ignore in headless environments


def save_checkpoint(model, optimizer, path):
    """
    Save a checkpoint.

    Args:
        model (nn.Module): Model.
        optimizer (Optimizer): Optimizer.
        path (str): Output file path.
    """
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)


def plot_metrics(history, out_path='training_curves.png'):
    """
    Plot train/val loss and accuracy and save to disk.

    Args:
        history (dict): Keys: train_loss, val_loss, train_acc, val_acc (lists).
        out_path (str): Output image file.
    """
    required_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Missing key '{key}' in history dict.")

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].grid(True, ls='--', alpha=0.4)
    axes[0].legend()

    # Accuracy subplot
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].grid(True, ls='--', alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved training curves to {out_path}")

def plot_confusion_matrix(model, dataloader, class_names, epoch, out_path='confusion_matrix.png'):
    """
    Compute and plot the confusion matrix for the validation set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Ensure all classes are represented
    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved confusion matrix to {out_path}_{epoch}")


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=50,
    use_amp=True,
    checkpoint_dir='checkpoints',
):
    """
    Train loop with mixed precision and metric history collection.

    Args:
        model (nn.Module): Model to train.
        dataloaders (dict): {'train': DataLoader, 'val': DataLoader}.
        dataset_sizes (dict): {'train': int, 'val': int}.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler.
        num_epochs (int): Number of epochs.
        use_amp (bool): Enable AMP.
        checkpoint_dir (str): Directory for checkpoints.

    Returns:
        tuple: (best_model, history_dict)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_weights = None
    start = time.time()
    print('Starting training...')
    # Metric history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    with TemporaryDirectory() as tmpdir:
        best_tmp_path = os.path.join(tmpdir, 'best_model.pt')

        for epoch in range(num_epochs):
            epoch_stats = {}

            for phase in ('train', 'val'):
                is_train = phase == 'train'
                print('phase = train' if is_train else 'phase = val')
                model.train(is_train)


                running_loss = 0.0
                running_corrects = 0

                loop = tqdm(dataloaders[phase],
                            desc=f'Epoch {epoch+1}/{num_epochs} [{phase}]',
                            leave=False)


                for inputs, labels in loop:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    print('inputs shape:', inputs.shape)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(is_train):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            outputs = model(inputs)
                            print('outputs shape:', outputs.shape)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        if is_train:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    print(f'loss: {loss.item():.4f}')
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

                if is_train and scheduler is not None:
                    scheduler.step()

                print(f'Epoch {epoch+1}/{num_epochs} | {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

                epoch_stats[f'{phase}_loss'] = epoch_loss
                print(f'epoch_stats: {epoch_stats}')
                epoch_stats[f'{phase}_acc'] = epoch_acc
                print(f'epoch_stats: {epoch_stats}')

                if not is_train and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_weights = model.state_dict()
                    torch.save(best_weights, best_tmp_path)
                    save_checkpoint(model, optimizer,
                                    os.path.join(checkpoint_dir, 'best_checkpoint.pth'))
            print(f'Best val Acc: {best_acc:.4f}')

            # Record metrics
            history['train_loss'].append(epoch_stats['train_loss'])
            history['val_loss'].append(epoch_stats['val_loss'])
            history['train_acc'].append(epoch_stats['train_acc'])
            history['val_acc'].append(epoch_stats['val_acc'])

            # Optional visualization of last batch
            if 'inputs' in locals():
                for i in [i for i in (0, 1, 2) if i < inputs.size(0)]:
                    imshow(inputs[i], f'val_sample_epoch{epoch+1}_idx{i}')
            plot_confusion_matrix(model, dataloaders['val'], class_names=['class0', 'class1'], epoch=epoch)

        elapsed = time.time() - start
        print(f'Training complete in {elapsed/60:.1f}m | Best val Acc: {best_acc:.4f}')

        if best_weights is not None:
            model.load_state_dict(best_weights)
    plot_metrics(history, f'training_curves_{num_epochs}_epochs_{model.fc.out_features}_classes.png')


    return model, history


def build_model(in_channels=84, num_classes=2, pretrained=True):
    """
    Build a ResNet101 with custom input channels and output classes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load ImageNet weights.

    Returns:
        nn.Module: Configured model.
    """
    model = models.resnet101(
        weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    )
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f'Built ResNet101 with {in_channels} input channels and {num_classes} output classes.')
    return model


def visualize_model(model, dataloader, num_images=4, out_prefix='viz'):
    """
    Save a few sample input composites.

    Args:
        model (nn.Module): Trained model (only used for device context).
        dataloader (DataLoader): Source of samples.
        num_images (int): Number of images to save.
        out_prefix (str): Filename prefix.
    """
    model.eval()
    shown = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            for i in range(inputs.size(0)):
                imshow(inputs[i], f'{out_prefix}_{shown}')
                shown += 1
                if shown >= num_images:
                    return


def main():
    """
    Orchestrate dataset loading, model training, visualization, and plotting.
    """
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataset = ClassifierDataset(
        '/home/ARO.local/tahor/PycharmProjects/data/pair_data',
        channels=840,
        c_step=10,
        transform=False,
        state='before'
    )
    print('Full dataset size:', len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=40, shuffle=True,
            num_workers=0, pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset, batch_size=10, shuffle=False,
            num_workers=0, pin_memory=True
        ),
    }
    print('Dataloaders ready.')

    model = build_model(in_channels=84, num_classes=2, pretrained=True)
    # if torch.cuda.device_count() > 1:
    #     print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
    #     model = nn.DataParallel(model)
    model = model.to(device)
    print('Model built to device.')

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print('Criterion, optimizer, scheduler set.')

    model, history = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=7,
        use_amp=True,
        checkpoint_dir='checkpoints'
    )

    visualize_model(model, dataloaders['val'], num_images=3)
    plot_metrics(history, f'training_curves_{1}_epochs_{2}_classes.png')

    torch.save(model.state_dict(), 'resnet101_classifier_final.pth')
    save_checkpoint(model, optimizer, 'checkpoints/final_checkpoint.pth')
    print('Saved final model and checkpoint.')


    print('Plotted training curves.')


if __name__ == '__main__':
    main()
