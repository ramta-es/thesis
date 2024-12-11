import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE1 = "cuda:1" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DEVICE = "cpu"
LEARNING_RATE = 4e-4
BATCH_SIZE = 2
NUM_WORKERS = 2
IMAGE_SIZE = 640
CHANNELS_IMG = 84
# MEAN = [0.5, 0.5, 0.5]
# STD = [0.5, 0.5, 0.5]
MEAN = 84 * [0.5]
STD = 84 * [0.5]
MAX_PIX_VAL = 4096.0
L1_LAMBDA = 100
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = 'disc.pth.tar'
CHECKPOINT_GEN = 'gen.pth.tar'

initial_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE)]
)


both_transforms = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD, max_pixel_value=MAX_PIX_VAL),
     A.HorizontalFlip(p=0.5), ],
    additional_targets={"image0": "image"}
)

transform_only_input = A.Compose(
    [
        # A.ColorJitter(p=0.1),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=MAX_PIX_VAL)#,
        # A.HorizontalFlip(p=0.5)
    ]
)
