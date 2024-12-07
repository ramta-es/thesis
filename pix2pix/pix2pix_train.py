import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import  Pix2PixDataset
from torch.utils.data import Subset
from pix2pix_model import Generator, Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Back, Style
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from paths.persimmon_paths import hpc_paths_for_data_after, hpc_paths_for_data_before
from sklearn.model_selection import train_test_split

# before_path = hpc_paths_for_data_before['single_fruit_path']
# after_path = hpc_paths_for_data_after['single_fruit_path']
test_dir = "/home/ARO.local/tahor/PycharmProjects/data/pair_data"


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler, writer, epoch):
    loop = tqdm(loader, leave=True)
    p = nn.AvgPool3d((10, 1, 1), stride=(10, 1, 1))

    for idx, (x, y) in enumerate(loop):
        # x = p(x)
        # y = p(y)
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        print('x shape train: ', x.shape)

        # Train Discriminator
        print(Fore.LIGHTGREEN_EX + 'train disc')
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            print("D_real_loss: ", D_real_loss)
            D_fake_loss = bce(D_real, torch.zeros_like(D_fake))
            print("D_fake_loss: ", D_fake_loss)
            D_loss = (D_real_loss + D_fake_loss) / 2
            print("D_loss: ", D_loss)

        writer.add_scalar('D_real_loss/idx', D_real_loss, idx)
        writer.add_scalar('D_fake_loss/idx', D_fake_loss, idx)
        writer.add_scalar('D_loss/idx', D_loss, idx)



        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        print(Fore.LIGHTRED_EX + 'train gen ')
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        writer.add_scalar('G_fake_loss/idx', G_fake_loss, idx)
        writer.add_scalar('DGloss/idx', G_loss, idx)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    writer.add_scalar('D_real_loss/epoch', D_real_loss, epoch)
    writer.add_scalar('D_fake_loss/epoch', D_fake_loss, epoch)
    writer.add_scalar('D_loss/epoch', D_loss, epoch)
    writer.add_scalar('G_fake_loss/epoch', G_fake_loss, epoch)
    writer.add_scalar('G_loss/epoch', G_loss, epoch)

    img_grid = torchvision.utils.make_grid(y_fake[:, [40, 30, 60], :, :])
    # print('image grid shape', np.ndarray(img_grid).shape)
    writer.add_image(f'HS_image_channel{[40, 30, 60]}', img_grid)
    writer.close()


def train_val_dataset(dataset, split: float) -> dict:
    train_idx, val_idx = train_test_split(range(len(dataset.before)), test_size=split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def main():
    disc = Discriminator(in_channels=84).to(config.DEVICE)
    gen = Generator(in_channels=84).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    ####################################################################################
    dataset = train_val_dataset(dataset=Pix2PixDataset(image_dir=test_dir, transform=True, channels=840, c_step=10), split=0.2)
    writer = SummaryWriter('runs/remote/pix2pix')
    ####################################################################################

    train_dataset = dataset['train']
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = dataset['val']
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOC {epoch}")
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, writer, epoch)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
