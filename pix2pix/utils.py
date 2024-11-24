import numpy as np
import torch
import config
from torchvision.utils import save_image


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        y_fake_copy = y_fake.cpu()
        x_copy = x.cpu()
        y_copy = y.cpu()
        np.save(folder + f"/y_gen_{epoch}.npy", y_fake_copy.numpy().squeeze().transpose(2, 1, 0))
        np.save(folder + f"/input_{epoch}.npy", x_copy.numpy().squeeze().transpose(2, 1, 0) * 0.5 + 0.5)
        if epoch == 1:
            np.save(folder + f"/label_{epoch}.npy", y_copy.numpy().squeeze().transpose(2, 1, 0) * 0.5 + 0.5)
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # ,and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
