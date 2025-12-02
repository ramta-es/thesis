
'''
import torch
from torch.utils.tensorboard import SummaryWriter

# Use a unique directory for logs
writer = SummaryWriter('/Users/ramtahor/PycharmProjects/thesis/resnet101_classifier/runs')

# Log scalar values
for i in range(100):
    writer.add_scalar('Loss/train', 0.1 * i, i)
    writer.add_scalar('Accuracy/train', 0.9 - 0.1 * i, i)

# Flush and close the writer
writer.flush()
writer.close()


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('/Users/ramtahor/PycharmProjects/thesis/resnet101_classifier/runs')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))


for i in range(100):
    loss = 0.1 * i
    writer.add_scalar('Loss/train', loss, i)
    acc = 0.9 - 0.1 * i
    writer.add_scalar('Accuracy/train', acc, i)

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

'''

import numpy as np
import matplotlib.pyplot as plt
image = np.load('/Volumes/Extreme Pro/test_for_viewer_np/reconstructed_normalize_correlation/reconstructedCorrelationNormalize.npy').astype(np.float32)
print(f"Original shape: {image.shape}")

# Transpose the array to rearrange dimensions
transposed_image = np.transpose(image, (1, 2, 0))
# Alternative: transposed_image = image.transpose(1, 2, 0)

# Print new shape
print(f"Transposed shape: {transposed_image.shape}")
print(np.sum(transposed_image[:, :, :], axis=2).shape)
summed_image = np.sum(transposed_image, axis=2)
normalized = (summed_image - np.min(summed_image)) / (np.max(summed_image) - np.min(summed_image))
plt.imshow(normalized, cmap='gray')
# Display the image (adjust indices as needed based on new shape)

plt.show()
