# Fine-tune ViT-H/14, RegNetY-32GF, ConvNeXt-Large, EfficientNet-B7
# Inputs: C x H x W = 84 x 640 x 640
# Outputs: num_classes = 4  OR  2  (both supported)

import math
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------ Config ------------------


model_names = [
    "vit_h_14",
    "regnet_y_32gf",
    "convnext_large",
    "efficientnet_b7",
]

in_channels = 84          # <-- updated channel count
num_classes = 4           # <-- set to 4 or 2
target_h, target_w = 640, 640
batch_size = 4
epochs = 10
lr = 1e-4
use_amp = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Helpers ------------------

class ConfigModel():
    def __init__(self, model_list, criterion, optimizer, scheduler, config_dict, scaler=None):
        self.model = model_list
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config_dict = config_dict
        self.scaler = scaler


    def pad_to_multiple(self, x: torch.Tensor, multiple: int = 14) -> torch.Tensor:
        """Pad H,W up to next multiple of 'multiple'. x: [B,C,H,W]"""
        _, _, H, W = x.shape
        pad_h = (multiple - (H % multiple)) % multiple
        pad_w = (multiple - (W % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)

    def set_input(self, module: nn.Module, in_ch: int) -> bool:  #replace_first_conv
        """
        Replace the first Conv2d expecting 3 channels to accept 'in_ch'.
        Weight init: channel-averaged from pretrained RGB.
        Returns True if a layer was replaced.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new_conv = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,             # keep original groups (usually 1)
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    w = child.weight  # [out, 3, k, k]
                    if in_ch == 3:
                        new_conv.weight.copy_(w)
                    else:
                        avg = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                        new_conv.weight.copy_(avg.repeat(1, in_ch, 1, 1) / math.sqrt(in_ch/3))
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(module, name, new_conv)
                return True
            if set_input(child, in_ch):
                return True
        return False

def set_output(model: nn.Module, num_classes: int, name: str): # replace_classifier_head
    n = name.lower()
    if n.startswith("vit"):
        # torchvision ViT has model.heads.head as final Linear
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            in_f = model.heads.head.in_features
            model.heads.head = nn.Linear(in_f, num_classes)
        else:
            for m in model.modules():
                if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 1000:
                    new = nn.Linear(m.in_features, num_classes)
                    m.weight = new.weight; m.bias = new.bias; m.out_features = num_classes
                    break
    elif n.startswith("convnext"):
        # model.classifier[-1] is final Linear
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif n.startswith("regnet"):
        # model.fc is final Linear
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
    elif n.startswith("efficientnet"):
        # model.classifier[-1] is final Linear
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        # generic fallback
        for m in model.modules():
            if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 1000:
                new = nn.Linear(m.in_features, num_classes)
                m.weight = new.weight; m.bias = new.bias; m.out_features = num_classes
                break

def load_model_generic(name: str, in_ch: int, num_classes: int) -> nn.Module:
    import torchvision.models as tvm

    builders = {
        "vit_h_14": tvm.vit_h_14,
        "regnet_y_32gf": tvm.regnet_y_32gf,
        "convnext_large": tvm.convnext_large,
        "efficientnet_b7": tvm.efficientnet_b7,
    }
    weights = {
        "vit_h_14": tvm.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        "regnet_y_32gf": tvm.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1,
        "convnext_large": tvm.ConvNeXt_Large_Weights.IMAGENET1K_V1,
        "efficientnet_b7": tvm.EfficientNet_B7_Weights.IMAGENET1K_V1,
    }

    if name not in builders:
        raise ValueError(f"Unsupported model: {name}")

    model = builders[name](weights=weights[name])

    # Adapt first conv (including ViT patch embed conv)
    if in_ch != 3:
        ok = set_input(model, in_ch)
        if not ok:
            # ViT patch embedding conv is model.conv_proj in torchvision
            if hasattr(model, "conv_proj") and isinstance(model.conv_proj, nn.Conv2d) and model.conv_proj.in_channels == 3:
                old = model.conv_proj
                new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=(old.bias is not None))
                with torch.no_grad():
                    w = old.weight
                    if in_ch == 3:
                        new.weight.copy_(w)
                    else:
                        avg = w.mean(dim=1, keepdim=True)
                        new.weight.copy_(avg.repeat(1, in_ch, 1, 1) / math.sqrt(in_ch/3))
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
                model.conv_proj = new

    set_output(model, num_classes, name)
    return model

# ------------------ Dummy dataset (replace with your real one) ------------------
class MyDataset(Dataset):
    def __init__(self, N=100):
        self.x = torch.randn(N, in_channels, target_h, target_w)
        self.y = torch.randint(0, num_classes, (N,))  # labels in [0, num_classes-1]

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ------------------ Train / Eval ------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Pad for ViT-H/14 (patch size 14)
        if hasattr(model, "conv_proj"):
            x = pad_to_multiple(x, 14)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if hasattr(model, "conv_proj"):
            x = pad_to_multiple(x, 14)
        out = model(x)
        loss = criterion(out, y)
        pred = out.argmax(1)
        tot_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n


def run(models: List[str], num_classes: int, image_dir: str, state: str = 'before'):
    """
    Args:
        models: List of model names to train
        num_classes: Number of output classes (2 or 4)
        image_dir: Path to your image directory
        state: 'before' or 'after' (which .npy files to load)
    """
    criterion = nn.CrossEntropyLoss()

    # Use your ClassifierDataset instead of MyDataset
    full_dataset = ClassifierDataset(
        image_dir=image_dir,
        channels=in_channels,  # 84 channels
        c_step=1,  # adjust based on your needs
        transform=False,  # set to True for training augmentation
        state=state
    )

    # Split into train/val (80/20 split)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(0.8 * dataset_size)

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    results = {}
    for name in models:
        print(f"\n=== Fine-tuning {name} (num_classes={num_classes}) ===")
        model = load_model_generic(name, in_channels, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

        best = 0.0
        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            vl, acc = evaluate(model, val_loader, criterion)
            print(f"[{name}] Epoch {epoch:02d} | train_loss={tr:.4f} | val_loss={vl:.4f} | val_acc={acc * 100:.2f}%")
            best = max(best, acc)
        results[name] = best
    return results


if __name__ == "__main__":
    # Specify your data directory
    image_dir = "/home/ARO.local/tahor/PycharmProjects/data/pair_data"

    # Run with 4 classes
    summary_4 = run(model_names, num_classes=4, image_dir=image_dir, state='before')
    print("\nBest val acc (4 classes):")
    for k, v in summary_4.items():
        print(f"  {k}: {v * 100:.2f}%")



    # Run with 2 classes
    summary_2 = run(model_names, num_classes=2)
    print("\nBest val acc (2 classes):")
    for k, v in summary_2.items():
        print(f"  {k}: {v*100:.2f}%")
