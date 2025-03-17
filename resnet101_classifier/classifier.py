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
# from dataset import ClassifierDataset
from resnet101_classifier.dataset import ClassifierDataset
from torch.utils.tensorboard import SummaryWriter
import clearml_agent
from tqdm import tqdm
import config


# Set CUDA_LAUNCH_BLOCKING to 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"CUDA_LAUNCH_BLOCKING is set to: {os.environ['CUDA_LAUNCH_BLOCKING']}")

cudnn.benchmark = True
# plt.ion()




###Visualize a few images###

def imshow(inp, fname: str, inp_channels=None, title=None):
    """Display image for Tensor."""
    if inp_channels is None:
        inp_channels = [44, 55, 60]
    inp = inp.numpy().transpose((1, 2, 0))[:, :, inp_channels]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.imsave(f'{fname}.png', inp)

    # plt.pause(0.001)  # pause a bit so that plots are updated

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)




# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
# print('classes', classes)


# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, 'out') #, title=[classes[x] for x in classes if len(classes) < 5])


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, train_writer, val_writer, num_epochs=25):
    since = time.time()
    print('line 74')
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        # torch.save(model.state_dict(), best_model_params_path)
        # save_checkpoint(model, optimizer)
        best_acc = 0.0
        print('model saved')
        for epoch in tqdm(range(num_epochs), desc='Tarin Model'):
            # print(f'Epoch {epoch}/{num_epochs - 1}')
            # print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(config.DEVICE)
                    labels = labels.to(config.DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / (dataset_sizes[phase])
                    if phase == 'train':
                        scheduler.step()
                    else:
                        epoch_acc = running_corrects.double() / (dataset_sizes[phase])



                        # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                        # deep copy the model

                        best_acc = epoch_acc

                        torch.save(model.state_dict(), best_model_params_path)
                        save_checkpoint(model, optimizer)
                if phase == 'train':
                    train_writer.add_scalar(f'loss', loss.item(), epoch)
                    train_writer.add_scalar('accuracy', epoch_acc, epoch)
                else:
                    val_writer.add_scalar(f'loss', loss.item(), epoch)
                    val_writer.add_scalar('accuracy', epoch_acc, epoch)

                train_writer.add_scalar(f'epoch_loss', epoch_loss, epoch)
            valid_indices = [i for i in [3, 5, 10] if i < inputs.size(0)]
            if valid_indices:
                img_grid = torchvision.utils.make_grid(inputs[valid_indices, :, :, :])
                val_writer.add_image('HS_image', img_grid)



    # img_grid = torchvision.utils.make_grid(inputs[[3, 5, 10], [40, 30, 60], :, :])
        # writer.add_image(f'HS_image_channel{40}', img_grid)
        # # writer.add_graph(torchvision.models.resnet101(False).cpu(), img_grid)
        # writer.close()



            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        val_writer.close()
        train_writer.close()
        return model





def visualize_model(model, dataloaders, train_writer, val_writer, num_images=1):
    was_training = model.training
    model.eval()
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title(f'predicted: {classes[preds[j]]}')
                imshow(inputs.cpu().data[j], 'inputs')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            print('inputs size:', (inputs.size()))
            # img_grid = torchvision.utils.make_grid(inputs.cpu()[:, 40, :, :])
            # print('image grid shape', np.ndarray(img_grid).shape)
            # writer.add_image(f'HS_image_channel{40}', img_grid)
            # writer.close()
            #
            # writer.add_graph(models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2), inputs.cpu())

    model.train(mode=was_training)


def main():
    # Data augmentation and normalization for training
    # Just normalization for validation

    dataset = ClassifierDataset('/home/ARO.local/tahor/PycharmProjects/data/pair_data', channels=840, c_step=10,
                                transform=False, state='before')

    print('line 204')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    data_dict = {'train': train_dataset, 'val': test_dataset}
    dataset_sizes = {x: len(data_dict[x]) for x in ['train', 'val']}

    print('line 221')

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True),
                   'val': torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)}

    ######
    valwriter = SummaryWriter('runs/remote/classifier')  ###### Added
    trainwriter = SummaryWriter('runs/remote/classifier')  ###### Added
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    model_ft.conv1 = nn.Conv2d(84, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           trainwriter, valwriter, num_epochs=100)


    visualize_model(model_ft, dataloaders, trainwriter, valwriter)
    print('vis model')
'''

# convnet as fix feature extractor####

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)




model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)



visualize_model(model_conv)

plt.ioff()
plt.show()



'''


###Inference on costume images####

'''
def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        # ax = plt.subplot(2,2,1)
        # ax.axis('off')
        # ax.set_title(f'Predicted: {preds[0]}')
        # imshow(img.cpu().data[0])

        model.train(mode=was_training)

'''

if __name__ == '__main__':
    main()










