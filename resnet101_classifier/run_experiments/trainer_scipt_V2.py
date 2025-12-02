"""
Generic Classification Training Framework
==========================================
Simplified version using direct DataLoader creation with ClassifierDataset.

Features:
- Hybrid model registry (pre-registered + auto-detection)
- Support for 640×640 + 84 channels (multi-spectral)
- Direct DataLoader usage (no config-based building)
- Smart optimizer/scheduler defaults with overrides
- Resume failed experiments
- Multi-GPU support (optional)
- Comprehensive error logging
- Automatic plot generation
"""

import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

import matplotlib
matplotlib.use('Agg')

import os
import sys
import yaml
import json
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import torchvision
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    vit_b_16, vit_b_32,
    convnext_tiny, convnext_small,
    mobilenet_v2, mobilenet_v3_small,
)

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score,
)

# Import your dataset
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from resnet101_classifier.dataset import ClassifierDataset


# ============================================================================
# MODEL REGISTRY & CONFIGURATION (unchanged)
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    model_fn: callable
    first_conv_path: str
    classifier_path: str
    architecture_family: str


class ModelRegistry:
    """Registry of pre-tested models with known layer paths."""

    _registry: Dict[str, ModelConfig] = {
        'resnet18': ModelConfig(resnet18, 'conv1', 'fc', 'cnn'),
        'resnet34': ModelConfig(resnet34, 'conv1', 'fc', 'cnn'),
        'resnet50': ModelConfig(resnet50, 'conv1', 'fc', 'cnn'),
        'resnet101': ModelConfig(resnet101, 'conv1', 'fc', 'cnn'),
        'resnet152': ModelConfig(resnet152, 'conv1', 'fc', 'cnn'),
        'efficientnet_b0': ModelConfig(efficientnet_b0, 'features.0.0', 'classifier.1', 'cnn'),
        'efficientnet_b1': ModelConfig(efficientnet_b1, 'features.0.0', 'classifier.1', 'cnn'),
        'efficientnet_b2': ModelConfig(efficientnet_b2, 'features.0.0', 'classifier.1', 'cnn'),
        'efficientnet_b3': ModelConfig(efficientnet_b3, 'features.0.0', 'classifier.1', 'cnn'),
        'vit_b_16': ModelConfig(vit_b_16, 'conv_proj', 'heads.head', 'transformer'),
        'vit_b_32': ModelConfig(vit_b_32, 'conv_proj', 'heads.head', 'transformer'),
        'convnext_tiny': ModelConfig(convnext_tiny, 'features.0.0', 'classifier.2', 'cnn'),
        'convnext_small': ModelConfig(convnext_small, 'features.0.0', 'classifier.2', 'cnn'),
        'mobilenet_v2': ModelConfig(mobilenet_v2, 'features.0.0', 'classifier.1', 'cnn'),
        'mobilenet_v3_small': ModelConfig(mobilenet_v3_small, 'features.0.0', 'classifier.3', 'cnn'),
    }

    @classmethod
    def get(cls, name: str) -> Optional[ModelConfig]:
        return cls._registry.get(name)

    @classmethod
    def register(cls, name: str, config: ModelConfig):
        cls._registry[name] = config

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._registry.keys())


# ============================================================================
# GENERIC MODEL BUILDER (unchanged)
# ============================================================================

class GenericModelBuilder:
    """Builds classification models with automatic adaptation."""

    def __init__(self):
        self.registry = ModelRegistry()

    def build(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build model with optimizer, scheduler, and criterion."""
        model_config = config['model']

        model = self._load_model(model_config)

        if model_config.get('input_channels', 3) != 3:
            model = self._adapt_input_channels(
                model, model_config['name'],
                model_config['input_channels'],
                model_config.get('channel_adaptation', 'repeat')
            )

        if model_config.get('input_size', (224, 224)) != (224, 224):
            model = self._adapt_spatial_size(
                model, model_config['name'],
                model_config['input_size']
            )

        model = self._replace_classifier(
            model, model_config['name'],
            model_config['num_classes']
        )

        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)
        criterion = nn.CrossEntropyLoss()

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
        }

    def _load_model(self, config: Dict[str, Any]) -> nn.Module:
        """Load model from source."""
        source = config['source']
        name = config['name']
        pretrained = config.get('pretrained', True)

        if source == 'torchvision':
            model_cfg = self.registry.get(name)
            if model_cfg is None:
                raise ValueError(f"Model '{name}' not in registry")

            model = model_cfg.model_fn(pretrained=pretrained)
            print(f"✓ Loaded {'pretrained' if pretrained else 'random init'} {name}")

        elif source == 'torch.hub':
            repo = config.get('repo', 'pytorch/vision:v0.10.0')
            model = torch.hub.load(repo, name, pretrained=pretrained)
            print(f"✓ Loaded {name} from torch.hub")

        elif source == 'custom':
            import importlib
            module = importlib.import_module(config['module_path'])
            model_class = getattr(module, config['class_name'])
            model = model_class(**config.get('init_kwargs', {}))
            print(f"✓ Loaded custom model: {config['class_name']}")

        else:
            raise ValueError(f"Unknown model source: {source}")

        return model

    def _adapt_input_channels(self, model: nn.Module, model_name: str,
                             in_channels: int, method: str = 'repeat') -> nn.Module:
        """Adapt first conv layer for arbitrary input channels."""
        model_cfg = self.registry.get(model_name)
        if model_cfg is None:
            conv_path = self._auto_detect_first_conv(model)
            print(f"⚠ Auto-detected first conv: {conv_path}")
        else:
            conv_path = model_cfg.first_conv_path

        parts = conv_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

        attr_name = parts[-1]
        old_conv = getattr(parent, attr_name) if not parts[-1].isdigit() else parent[int(parts[-1])]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        with torch.no_grad():
            if method == 'repeat':
                old_weight = old_conv.weight
                repeat_factor = in_channels // 3
                remainder = in_channels % 3

                if repeat_factor > 0:
                    repeated = old_weight.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        extra = old_weight[:, :remainder, :, :]
                        new_conv.weight.copy_(torch.cat([repeated, extra], dim=1))
                    else:
                        new_conv.weight.copy_(repeated)
                else:
                    new_conv.weight.copy_(old_weight[:, :in_channels, :, :])

            elif method == 'random':
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        if not parts[-1].isdigit():
            setattr(parent, attr_name, new_conv)
        else:
            parent[int(parts[-1])] = new_conv

        print(f"✓ Adapted input channels: 3 → {in_channels} (method: {method})")
        return model

    def _adapt_spatial_size(self, model: nn.Module, model_name: str,
                           input_size: Tuple[int, int]) -> nn.Module:
        """Adapt Vision Transformer models to arbitrary spatial size."""
        model_cfg = self.registry.get(model_name)
        if not (model_cfg and model_cfg.architecture_family == 'transformer'):
            return model

        h, w = input_size
        if h != w:
            raise ValueError(f"ViT requires square inputs, got {input_size}")

        old_img_size = getattr(model, 'image_size', 224)
        if old_img_size == h:
            return model

        model.image_size = h
        patch_embed = model.conv_proj
        patch_size = patch_embed.kernel_size[0]
        num_patches_new = (h // patch_size) * (w // patch_size)

        with torch.no_grad():
            pos_embed = model.encoder.pos_embedding
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]

            num_patches_old = patch_pos.shape[1]
            old_grid_size = int(num_patches_old ** 0.5)
            embed_dim = patch_pos.shape[2]

            patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, embed_dim)
            patch_pos = patch_pos.permute(0, 3, 1, 2)

            new_grid_size = h // patch_size
            patch_pos_resized = F.interpolate(
                patch_pos, size=(new_grid_size, new_grid_size),
                mode='bicubic', align_corners=False
            )

            patch_pos_resized = patch_pos_resized.permute(0, 2, 3, 1)
            patch_pos_resized = patch_pos_resized.reshape(1, num_patches_new, embed_dim)
            new_pos_embed = torch.cat([cls_pos, patch_pos_resized], dim=1)
            model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

        print(f"✓ Adapted spatial size for ViT: {old_img_size}×{old_img_size} → {h}×{w}")
        return model

    def _replace_classifier(self, model: nn.Module, model_name: str,
                           num_classes: int) -> nn.Module:
        """Replace classifier layer."""
        model_cfg = self.registry.get(model_name)
        if model_cfg is None:
            classifier_path = self._auto_detect_classifier(model)
            print(f"⚠ Auto-detected classifier: {classifier_path}")
        else:
            classifier_path = model_cfg.classifier_path

        parts = classifier_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

        attr_name = parts[-1]
        old_classifier = getattr(parent, attr_name) if not parts[-1].isdigit() else parent[int(parts[-1])]

        if isinstance(old_classifier, nn.Linear):
            in_features = old_classifier.in_features
        elif isinstance(old_classifier, nn.Sequential):
            for layer in reversed(old_classifier):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        else:
            raise ValueError(f"Unsupported classifier type: {type(old_classifier)}")

        new_classifier = nn.Linear(in_features, num_classes)

        if not parts[-1].isdigit():
            setattr(parent, attr_name, new_classifier)
        else:
            parent[int(parts[-1])] = new_classifier

        print(f"✓ Replaced classifier: {in_features} → {num_classes} classes")
        return model

    def _auto_detect_first_conv(self, model: nn.Module) -> str:
        """Auto-detect first convolutional layer."""
        for path in ['conv1', 'features.0.0', 'conv_proj', 'stem.0']:
            try:
                parts = path.split('.')
                obj = model
                for part in parts:
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                if isinstance(obj, nn.Conv2d):
                    return path
            except:
                continue
        raise ValueError("Could not auto-detect first conv layer")

    def _auto_detect_classifier(self, model: nn.Module) -> str:
        """Auto-detect classifier layer."""
        for path in ['fc', 'classifier', 'head', 'heads.head', 'classifier.1']:
            try:
                parts = path.split('.')
                obj = model
                for part in parts:
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                if isinstance(obj, (nn.Linear, nn.Sequential)):
                    return path
            except:
                continue
        raise ValueError("Could not auto-detect classifier layer")

    def _create_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer with smart defaults."""
        training_config = config.get('training', {})
        model_config = config['model']

        model_cfg = self.registry.get(model_config['name'])
        arch_family = model_cfg.architecture_family if model_cfg else 'cnn'

        if arch_family == 'cnn':
            default_optimizer = 'sgd'
            default_lr = 0.01
            default_momentum = 0.9
        else:
            default_optimizer = 'adamw'
            default_lr = 3e-4
            default_momentum = None

        optimizer_name = training_config.get('optimizer', default_optimizer)
        lr = training_config.get('lr', default_lr)
        weight_decay = training_config.get('weight_decay', 1e-4)

        if optimizer_name.lower() == 'sgd':
            momentum = training_config.get('momentum', default_momentum)
            optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        print(f"✓ Created optimizer: {optimizer_name.upper()} (lr={lr})")
        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer,
                         config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        training_config = config.get('training', {})
        model_config = config['model']

        model_cfg = self.registry.get(model_config['name'])
        arch_family = model_cfg.architecture_family if model_cfg else 'cnn'

        default_scheduler = 'step' if arch_family == 'cnn' else 'cosine'
        scheduler_name = training_config.get('scheduler', default_scheduler)

        if scheduler_name.lower() == 'step':
            step_size = training_config.get('step_size', 10)
            gamma = training_config.get('gamma', 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'cosine':
            t_max = training_config.get('t_max', training_config.get('num_epochs', 50))
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        elif scheduler_name.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        print(f"✓ Created scheduler: {scheduler_name}")
        return scheduler


# ============================================================================
# GENERAL TRAINER (modified to only accept DataLoaders)
# ============================================================================

class GeneralTrainer:
    """Generic trainer accepting only DataLoader objects."""

    def __init__(self, device: str = 'cuda', use_amp: bool = True,
                 grad_accumulation_steps: int = 1, grad_clip: float = 1.0,
                 early_stopping_patience: int = 10):
        self.device = device
        self.use_amp = use_amp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience

        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_f2s = []
        self.val_roc_aucs = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train(self, model_dict: Dict[str, Any], train_loader: DataLoader,
              val_loader: DataLoader, config: Dict[str, Any],
              output_dir: str) -> Dict[str, Any]:
        """Train model - now only accepts DataLoader objects."""
        import time
        start_time = time.time()

        # Extract components
        model = model_dict['model']
        optimizer = model_dict['optimizer']
        scheduler = model_dict['scheduler']
        criterion = model_dict['criterion']

        # Setup model
        self.model = self._setup_model(model)
        criterion = criterion.to(self.device)

        # Training parameters
        num_epochs = config['training']['num_epochs']
        num_classes = config['model']['num_classes']

        print(f"\n{'='*60}")
        print(f"Starting Training: {config['name']}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Use AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        try:
            for epoch in range(num_epochs):
                # Train
                train_loss, train_acc = self._train_epoch(
                    self.model, train_loader, criterion, optimizer, epoch, num_epochs
                )

                # Validate
                val_loss, val_acc, y_true, y_pred, y_prob = self._validate_epoch(
                    self.model, val_loader, criterion, epoch, num_epochs
                )

                # Update scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Calculate advanced metrics
                y_true_np = np.array(y_true)
                y_pred_np = np.array(y_pred)

                precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                f2 = fbeta_score(y_true_np, y_pred_np, beta=2, average='macro', zero_division=0)

                # ROC AUC
                try:
                    unique_classes = np.unique(y_true_np)
                    if len(unique_classes) < num_classes:
                        y_prob_subset = y_prob[:, unique_classes]
                        roc_auc = roc_auc_score(y_true_np, y_prob_subset, multi_class='ovr', average='macro')
                    else:
                        roc_auc = roc_auc_score(y_true_np, y_prob, multi_class='ovr', average='macro')
                except Exception as e:
                    print(f"\nWarning: Could not calculate ROC AUC: {e}")
                    roc_auc = 0.0

                # Track metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                self.val_precisions.append(precision)
                self.val_recalls.append(recall)
                self.val_f1s.append(f1)
                self.val_f2s.append(f2)
                self.val_roc_aucs.append(roc_auc)

                # Save best checkpoint
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(self.model, optimizer, scheduler, epoch, output_dir, is_best=True)
                    print(f"✓ New best model saved (val_acc: {val_acc:.2f}%)")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

                # Periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(self.model, optimizer, scheduler, epoch, output_dir, is_best=False)

            # Generate plots
            self._plot_training_curves(output_dir)
            self._plot_advanced_metrics(output_dir)
            self._plot_confusion_matrix(y_true, y_pred, output_dir)

            # Save classification report
            report = classification_report(y_true, y_pred, zero_division=0)
            report_path = os.path.join(output_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)

            # Save experiment summary
            end_time = time.time()
            self._save_experiment_summary(
                config, train_loader, val_loader, output_dir,
                start_time, end_time, status="SUCCESS"
            )

            print(f"\n✓ Training completed!")
            print(f"✓ Best val accuracy: {self.best_val_acc:.4f}")
            print(f"✓ Outputs saved to: {output_dir}\n")

        except Exception as e:
            end_time = time.time()
            try:
                self._save_experiment_summary(
                    config, train_loader, val_loader, output_dir,
                    start_time, end_time, status="FAILED"
                )
            except Exception as summary_error:
                print(f"Warning: Could not save experiment summary: {summary_error}")
            raise e

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1s': self.val_f1s,
            'val_f2s': self.val_f2s,
            'val_roc_aucs': self.val_roc_aucs,
            'best_val_acc': self.best_val_acc,
        }

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training."""
        model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        return model

    def _train_epoch(self, model: nn.Module, loader: DataLoader,
                     criterion: nn.Module, optimizer: torch.optim.Optimizer,
                     epoch: int, num_epochs: int) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.grad_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.use_amp:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': acc})

        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       criterion: nn.Module, epoch: int,
                       num_epochs: int) -> Tuple[float, float, List[int], List[int], np.ndarray]:
        """Validate for one epoch."""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_prob_list = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                probs = F.softmax(outputs, dim=1)
                y_prob_list.append(probs.cpu().numpy())

                current_loss = val_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({'loss': current_loss, 'acc': current_acc})

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        y_prob = np.vstack(y_prob_list)

        return val_loss, val_acc, y_true, y_pred, y_prob

    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler._LRScheduler,
                        epoch: int, output_dir: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }

        path = os.path.join(output_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)

    def _plot_training_curves(self, output_dir: str):
        """Plot loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], output_dir: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    def _plot_advanced_metrics(self, output_dir: str):
        """Plot precision, recall, F1, F2."""
        if len(self.val_precisions) == 0:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(self.val_precisions) + 1)

        ax1.plot(epochs, self.val_precisions, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Precision (macro)', fontsize=12)
        ax1.set_title('Validation Precision', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        ax2.plot(epochs, self.val_recalls, marker='o', color='orange', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Recall (macro)', fontsize=12)
        ax2.set_title('Validation Recall', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        ax3.plot(epochs, self.val_f1s, marker='o', color='green', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('F1 (macro)', fontsize=12)
        ax3.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        ax4.plot(epochs, self.val_f2s, marker='o', color='red', linewidth=2, markersize=6)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('F2 (macro)', fontsize=12)
        ax4.set_title('Validation F2 Score', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'advanced_metrics.png'), dpi=150)
        plt.close()

    def _save_experiment_summary(self, config: Dict[str, Any],
                                train_loader: DataLoader, val_loader: DataLoader,
                                output_dir: str, start_time: float,
                                end_time: float, status: str = "SUCCESS"):
        """Save comprehensive experiment summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary = {
            "experiment_name": config['name'],
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "status": status,
            "duration_seconds": round(end_time - start_time, 2),
            "device": str(self.device),
            "use_amp": self.use_amp,
            "early_stopping_triggered": self.epochs_without_improvement >= self.early_stopping_patience,
            "config": config,
            "dataset_stats": {
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "num_classes": config['model']['num_classes'],
            },
            "results": {
                "best_val_acc": float(self.best_val_acc),
                "best_epoch": int(self.train_losses.index(min(self.train_losses)) + 1) if self.train_losses else 0,
                "final_train_loss": float(self.train_losses[-1]) if self.train_losses else None,
                "final_val_loss": float(self.val_losses[-1]) if self.val_losses else None,
                "final_train_acc": float(self.train_accs[-1]) if self.train_accs else None,
                "final_val_acc": float(self.val_accs[-1]) if self.val_accs else None,
                "final_precision": float(self.val_precisions[-1]) if self.val_precisions else None,
                "final_recall": float(self.val_recalls[-1]) if self.val_recalls else None,
                "final_f1": float(self.val_f1s[-1]) if self.val_f1s else None,
                "final_f2": float(self.val_f2s[-1]) if self.val_f2s else None,
                "final_roc_auc": float(self.val_roc_aucs[-1]) if self.val_roc_aucs else None
            },
            "training_history": {
                "train_losses": [float(x) for x in self.train_losses],
                "val_losses": [float(x) for x in self.val_losses],
                "train_accs": [float(x) for x in self.train_accs],
                "val_accs": [float(x) for x in self.val_accs],
                "val_precisions": [float(x) for x in self.val_precisions],
                "val_recalls": [float(x) for x in self.val_recalls],
                "val_f1s": [float(x) for x in self.val_f1s],
                "val_f2s": [float(x) for x in self.val_f2s],
                "val_roc_aucs": [float(x) for x in self.val_roc_aucs]
            },
            "model_info": {
                "architecture": config['model']['name'],
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "input_shape": [config['model']['input_channels'],
                               config['model']['input_size'][0],
                               config['model']['input_size'][1]],
                "output_classes": config['model']['num_classes'],
                "checkpoint_path": "best_model.pth"
            }
        }

        summary_path = os.path.join(output_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved experiment summary: {summary_path}")


# ============================================================================
# EXPERIMENT RUNNER (modified to create DataLoaders in main)
# ============================================================================

class ExperimentRunner:
    """Orchestrates experiments with error handling."""

    def __init__(self, base_output_dir: str = 'runs'):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        self.results = {}

    def run_experiments(self, experiments: List[Dict[str, Any]]):
        """Run all experiments."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RUNNER")
        print(f"{'='*60}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"{'='*60}\n")

        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] Running: {exp['name']}")
            success = self._run_single_experiment(exp)
            self.results[exp['name']] = 'SUCCESS' if success else 'FAILED'

        self._generate_summary_report()

    def _run_single_experiment(self, experiment: Dict[str, Any]) -> bool:
        """Run single experiment."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{experiment['name']}_{timestamp}"
        output_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            config_path = os.path.join(output_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(experiment['config'], f, default_flow_style=False)

            # Build model
            builder = GenericModelBuilder()
            model_dict = builder.build(experiment['config'])

            # Create trainer
            training_config = experiment['config']['training']
            trainer = GeneralTrainer(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_amp=training_config.get('use_amp', True),
                grad_accumulation_steps=training_config.get('grad_accumulation_steps', 1),
                grad_clip=training_config.get('grad_clip', 1.0),
                early_stopping_patience=training_config.get('early_stopping_patience', 10)
            )

            # Train with pre-built DataLoaders
            history = trainer.train(
                model_dict=model_dict,
                train_loader=experiment['train_loader'],
                val_loader=experiment['val_loader'],
                config=experiment['config'],
                output_dir=output_dir
            )

            history_path = os.path.join(output_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

            print(f"✓ Experiment completed: {experiment['name']}")
            return True

        except Exception as e:
            print(f"✗ Experiment failed: {experiment['name']}")
            print(f"Error: {str(e)}")
            trace = traceback.format_exc()
            self._save_error(experiment['config'], e, trace, output_dir)
            return False

    def _save_error(self, config: Dict[str, Any], exception: Exception,
                   trace: str, output_dir: str):
        """Save error details."""
        error_log = f"""
{'='*60}
EXPERIMENT FAILED
{'='*60}
Name: {config['name']}
Timestamp: {datetime.now().isoformat()}

Exception Type: {type(exception).__name__}
Exception Message: {str(exception)}

Stack Trace:
{trace}

Configuration:
{yaml.dump(config, default_flow_style=False)}
{'='*60}
"""
        error_path = os.path.join(output_dir, 'errors.txt')
        with open(error_path, 'w') as f:
            f.write(error_log)

    def _generate_summary_report(self):
        """Generate summary report."""
        report_path = os.path.join(self.base_output_dir, 'summary_report.txt')

        report = f"""
{'='*60}
EXPERIMENT SUMMARY
{'='*60}
Total Experiments: {len(self.results)}
Successful: {sum(1 for v in self.results.values() if v == 'SUCCESS')}
Failed: {sum(1 for v in self.results.values() if v == 'FAILED')}

Results:
"""
        for exp_name, status in self.results.items():
            report += f"  {exp_name:50s} {status}\n"

        report += f"{'='*60}\n"

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n{report}")
        print(f"✓ Summary report saved to {report_path}")


# ============================================================================
# MAIN - Create DataLoaders here
# ============================================================================

def create_dataloaders(dataset_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val DataLoaders from config.

    This is where you manually create your datasets and loaders.
    """
    # Extract parameters
    image_dir = dataset_config['path']
    channels = dataset_config['channels']
    c_step = dataset_config['c_step']
    state = dataset_config['state']
    label_type = dataset_config['label_type']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']

    print(f"\nCreating dataloaders:")
    print(f"  Image dir: {image_dir}")
    print(f"  Channels: {channels} (step: {c_step})")
    print(f"  State: {state}")
    print(f"  Label type: {label_type}")
    print(f"  Batch size: {batch_size}")

    # Create train dataset
    train_dataset = ClassifierDataset(
        image_dir=image_dir,
        channels=channels,
        c_step=c_step,
        transform=True,  # Augmentation for training
        state=state,
        label_type=label_type
    )

    # Create val dataset
    val_dataset = ClassifierDataset(
        image_dir=image_dir,
        channels=channels,
        c_step=c_step,
        transform=False,  # No augmentation for validation
        state=state,
        label_type=label_type
    )

    # Split indices 80/20
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    indices = list(range(dataset_size))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"✓ Created train loader: {len(train_subset)} samples")
    print(f"✓ Created val loader: {len(val_subset)} samples\n")

    return train_loader, val_loader


def main():
    """Main entry point - define experiments with DataLoaders."""

    # Define dataset config (used to create loaders)
    dataset_config = {
        'path': '/home/ARO.local/tahor/PycharmProjects/data/pair_data',
        'channels': 840,
        'c_step': 10,
        'state': 'before',
        'label_type': 'ABCD',
        'batch_size': 10,
        'num_workers': 4,
    }

    # Create dataloaders once
    train_loader, val_loader = create_dataloaders(dataset_config)

    # Define experiments - now with pre-built loaders
    EXPERIMENTS = [
        {
            'name': 'exp1_resnet101',
            'config': {
                'name': 'exp1_resnet101',
                'model': {
                    'source': 'torchvision',
                    'name': 'resnet101',
                    'pretrained': True,
                    'input_channels': 84,
                    'input_size': (640, 640),
                    'channel_adaptation': 'repeat',
                    'num_classes': 4,
                },
                'training': {
                    'num_epochs': 3,
                    'optimizer': 'adamw',
                    'lr': 1e-3,
                    'weight_decay': 1e-4,
                    'scheduler': 'step',
                    'step_size': 10,
                    'gamma': 0.1,
                    'use_amp': True,
                    'grad_accumulation_steps': 1,
                    'grad_clip': 1.0,
                    'early_stopping_patience': 10,
                },
            },
            'train_loader': train_loader,
            'val_loader': val_loader,
        },

        {
            'name': 'exp2_vit_b_16',
            'config': {
                'name': 'exp2_vit_b_16',
                'model': {
                    'source': 'torchvision',
                    'name': 'vit_b_16',
                    'pretrained': True,
                    'input_channels': 84,
                    'input_size': (640, 640),
                    'channel_adaptation': 'repeat',
                    'num_classes': 4,
                },
                'training': {
                    'num_epochs': 3,
                    'optimizer': 'adamw',
                    'lr': 1e-3,
                    'weight_decay': 1e-4,
                    'scheduler': 'cosine',
                    't_max': 3,
                    'use_amp': True,
                    'grad_accumulation_steps': 1,
                    'grad_clip': 1.0,
                    'early_stopping_patience': 10,
                },
            },
            'train_loader': train_loader,
            'val_loader': val_loader,
        },

        {
            'name': 'exp3_efficientnet_b0',
            'config': {
                'name': 'exp3_efficientnet_b0',
                'model': {
                    'source': 'torchvision',
                    'name': 'efficientnet_b0',
                    'pretrained': True,
                    'input_channels': 84,
                    'input_size': (640, 640),
                    'channel_adaptation': 'repeat',
                    'num_classes': 4,
                },
                'training': {
                    'num_epochs': 3,
                    'optimizer': 'adamw',
                    'lr': 1e-3,
                    'weight_decay': 1e-4,
                    'scheduler': 'step',
                    'step_size': 10,
                    'gamma': 0.1,
                    'use_amp': True,
                    'grad_accumulation_steps': 1,
                    'grad_clip': 1.0,
                    'early_stopping_patience': 10,
                },
            },
            'train_loader': train_loader,
            'val_loader': val_loader,
        },
    ]

    # Run experiments
    runner = ExperimentRunner(base_output_dir='runs')
    runner.run_experiments(EXPERIMENTS)


if __name__ == '__main__':
    main()