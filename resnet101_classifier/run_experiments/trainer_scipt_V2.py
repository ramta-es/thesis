"""
Generic Classification Training Framework
==========================================
A modular system for training any classification model on any dataset.

Features:
- Hybrid model registry (pre-registered + auto-detection)
- Support for 640×640 + 84 channels (multi-spectral)
- Both dataloader options (pre-built or config-based)
- Smart optimizer/scheduler defaults with overrides
- Resume failed experiments
- Multi-GPU support (optional)
- Comprehensive error logging
- Automatic plot generation (loss, accuracy, confusion matrix)
"""

import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import os
import sys
import yaml
import json
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import torchvision
from torchvision import transforms, datasets
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32,
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ============================================================================
# MODEL REGISTRY & CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    model_fn: callable
    first_conv_path: str
    classifier_path: str
    architecture_family: str  # 'cnn' or 'transformer'


class ModelRegistry:
    """Registry of pre-tested models with known layer paths."""

    _registry: Dict[str, ModelConfig] = {
        # ResNet family
        'resnet18': ModelConfig(
            model_fn=resnet18,
            first_conv_path='conv1',
            classifier_path='fc',
            architecture_family='cnn'
        ),
        'resnet34': ModelConfig(
            model_fn=resnet34,
            first_conv_path='conv1',
            classifier_path='fc',
            architecture_family='cnn'
        ),
        'resnet50': ModelConfig(
            model_fn=resnet50,
            first_conv_path='conv1',
            classifier_path='fc',
            architecture_family='cnn'
        ),
        'resnet101': ModelConfig(
            model_fn=resnet101,
            first_conv_path='conv1',
            classifier_path='fc',
            architecture_family='cnn'
        ),
        'resnet152': ModelConfig(
            model_fn=resnet152,
            first_conv_path='conv1',
            classifier_path='fc',
            architecture_family='cnn'
        ),

        # EfficientNet family
        'efficientnet_b0': ModelConfig(
            model_fn=efficientnet_b0,
            first_conv_path='features.0.0',
            classifier_path='classifier.1',
            architecture_family='cnn'
        ),
        'efficientnet_b1': ModelConfig(
            model_fn=efficientnet_b1,
            first_conv_path='features.0.0',
            classifier_path='classifier.1',
            architecture_family='cnn'
        ),
        'efficientnet_b2': ModelConfig(
            model_fn=efficientnet_b2,
            first_conv_path='features.0.0',
            classifier_path='classifier.1',
            architecture_family='cnn'
        ),
        'efficientnet_b3': ModelConfig(
            model_fn=efficientnet_b3,
            first_conv_path='features.0.0',
            classifier_path='classifier.1',
            architecture_family='cnn'
        ),

        # Vision Transformers
        'vit_b_16': ModelConfig(
            model_fn=vit_b_16,
            first_conv_path='conv_proj',
            classifier_path='heads.head',
            architecture_family='transformer'
        ),
        'vit_b_32': ModelConfig(
            model_fn=vit_b_32,
            first_conv_path='conv_proj',
            classifier_path='heads.head',
            architecture_family='transformer'
        ),

        # ConvNeXt family
        'convnext_tiny': ModelConfig(
            model_fn=convnext_tiny,
            first_conv_path='features.0.0',
            classifier_path='classifier.2',
            architecture_family='cnn'
        ),
        'convnext_small': ModelConfig(
            model_fn=convnext_small,
            first_conv_path='features.0.0',
            classifier_path='classifier.2',
            architecture_family='cnn'
        ),

        # MobileNet family
        'mobilenet_v2': ModelConfig(
            model_fn=mobilenet_v2,
            first_conv_path='features.0.0',
            classifier_path='classifier.1',
            architecture_family='cnn'
        ),
        'mobilenet_v3_small': ModelConfig(
            model_fn=mobilenet_v3_small,
            first_conv_path='features.0.0',
            classifier_path='classifier.3',
            architecture_family='cnn'
        ),
    }

    @classmethod
    def get(cls, name: str) -> Optional[ModelConfig]:
        """Get model config from registry."""
        return cls._registry.get(name)

    @classmethod
    def register(cls, name: str, config: ModelConfig):
        """Register a new model."""
        cls._registry[name] = config

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._registry.keys())


# ============================================================================
# GENERIC MODEL BUILDER
# ============================================================================

class GenericModelBuilder:
    """
    Builds classification models with automatic adaptation for:
    - Arbitrary input channels (e.g., 84 for multi-spectral)
    - Arbitrary spatial sizes (e.g., 640×640)
    - Custom number of classes
    - Pretrained weights initialization
    """

    def __init__(self):
        self.registry = ModelRegistry()

    def build(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build model with optimizer, scheduler, and criterion.

        Args:
            config: Model configuration dict

        Returns:
            dict: Contains model, optimizer, scheduler, criterion
        """
        model_config = config['model']

        # Load base model
        model = self._load_model(model_config)

        # Adapt input channels if needed
        if model_config.get('input_channels', 3) != 3:
            model = self._adapt_input_channels(
                model,
                model_config['name'],
                model_config['input_channels'],
                model_config.get('channel_adaptation', 'repeat')
            )

        # Adapt spatial size for Vision Transformers
        if model_config.get('input_size', (224, 224)) != (224, 224):
            model = self._adapt_spatial_size(
                model,
                model_config['name'],
                model_config['input_size']
            )

        # Replace classifier layer
        model = self._replace_classifier(
            model,
            model_config['name'],
            model_config['num_classes']
        )

        # Create optimizer
        optimizer = self._create_optimizer(model, config)

        # Create scheduler
        scheduler = self._create_scheduler(optimizer, config)

        # Create criterion
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
            # Check registry first
            model_cfg = self.registry.get(name)
            if model_cfg is None:
                raise ValueError(f"Model '{name}' not in registry. Please register it first.")

            # Load model
            if pretrained:
                model = model_cfg.model_fn(pretrained=True)
                print(f"✓ Loaded pretrained {name}")
            else:
                model = model_cfg.model_fn(pretrained=False)
                print(f"✓ Loaded {name} (random init)")

        elif source == 'torch.hub':
            # Load from Torch Hub
            repo = config.get('repo', 'pytorch/vision:v0.10.0')
            model = torch.hub.load(repo, name, pretrained=pretrained)
            print(f"✓ Loaded {name} from torch.hub")

        elif source == 'custom':
            # Load custom model (user must provide import path)
            module_path = config['module_path']
            class_name = config['class_name']
            # Dynamic import
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            model = model_class(**config.get('init_kwargs', {}))
            print(f"✓ Loaded custom model: {class_name}")

        else:
            raise ValueError(f"Unknown model source: {source}")

        return model

    def _adapt_input_channels(
        self,
        model: nn.Module,
        model_name: str,
        in_channels: int,
        method: str = 'repeat'
    ) -> nn.Module:
        """
        Adapt first convolutional layer for arbitrary input channels.

        Args:
            model: The model to adapt
            model_name: Name of the model
            in_channels: Target number of input channels
            method: 'repeat' or 'random'
        """
        # Get first conv layer path from registry
        model_cfg = self.registry.get(model_name)
        if model_cfg is None:
            conv_path = self._auto_detect_first_conv(model)
            print(f"⚠ Auto-detected first conv: {conv_path}")
        else:
            conv_path = model_cfg.first_conv_path

        # Navigate to the layer
        parts = conv_path.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        old_conv = getattr(parent, attr_name) if not parts[-1].isdigit() else parent[int(parts[-1])]

        # Create new conv layer
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        # Initialize weights based on method
        with torch.no_grad():
            if method == 'repeat':
                # Repeat old weights across new channels
                old_weight = old_conv.weight  # [out_ch, 3, k, k]
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
                    # If in_channels < 3, just take first channels
                    new_conv.weight.copy_(old_weight[:, :in_channels, :, :])

            elif method == 'random':
                # Random initialization
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            # Copy bias if exists
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace old conv with new conv
        if not parts[-1].isdigit():
            setattr(parent, attr_name, new_conv)
        else:
            parent[int(parts[-1])] = new_conv

        print(f"✓ Adapted input channels: 3 → {in_channels} (method: {method})")
        return model

    def _adapt_spatial_size(
            self,
            model: nn.Module,
            model_name: str,
            input_size: Tuple[int, int]
    ) -> nn.Module:
        """
        Adapt Vision Transformer models to arbitrary input spatial size.

        For torchvision ViT:
        - Updates `image_size`
        - Resizes positional embeddings via interpolation

        For CNNs: No action needed (adaptive pooling handles it).
        """
        model_cfg = self.registry.get(model_name)
        if not (model_cfg and model_cfg.architecture_family == 'transformer'):
            return model

        h, w = input_size
        if h != w:
            raise ValueError(f"ViT requires square inputs, got {input_size}")

        old_img_size = getattr(model, 'image_size', 224)
        if old_img_size == h:
            return model

        # Update the image_size attribute that _process_input checks
        model.image_size = h

        # Get patch size from conv_proj layer
        patch_embed = model.conv_proj
        patch_size = patch_embed.kernel_size[0]

        # Calculate new number of patches
        num_patches_new = (h // patch_size) * (w // patch_size)

        # Resize positional embeddings
        with torch.no_grad():
            pos_embed = model.encoder.pos_embedding  # [1, 1 + num_patches_old, dim]
            cls_pos = pos_embed[:, :1, :]  # class token
            patch_pos = pos_embed[:, 1:, :]  # patch positions

            # Derive old grid size
            num_patches_old = patch_pos.shape[1]
            old_grid_size = int(num_patches_old ** 0.5)

            # Reshape to 2D grid
            embed_dim = patch_pos.shape[2]
            patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, embed_dim)
            patch_pos = patch_pos.permute(0, 3, 1, 2)  # [1, dim, H, W]

            # Interpolate to new grid size
            new_grid_size = h // patch_size
            patch_pos_resized = F.interpolate(
                patch_pos,
                size=(new_grid_size, new_grid_size),
                mode='bicubic',
                align_corners=False,
            )

            # Reshape back
            patch_pos_resized = patch_pos_resized.permute(0, 2, 3, 1)
            patch_pos_resized = patch_pos_resized.reshape(1, num_patches_new, embed_dim)

            # Concatenate class token and resized patch embeddings
            new_pos_embed = torch.cat([cls_pos, patch_pos_resized], dim=1)
            model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

        print(f"✓ Adapted spatial size for ViT: ({old_img_size}, {old_img_size}) → ({h}, {w})")
        return model

    # def _interpolate_pos_embeddings(
    #     self,
    #     model: nn.Module,
    #     input_size: Tuple[int, int]
    # ):
    #     """Interpolate position embeddings for ViT."""
    #     # Get current position embeddings
    #     pos_embed = model.encoder.pos_embedding  # Shape: [1, num_patches + 1, embed_dim]
    #
    #     # Calculate current and target grid sizes
    #     num_patches = pos_embed.shape[1] - 1  # Exclude class token
    #     old_grid_size = int(np.sqrt(num_patches))
    #     new_grid_size = input_size[0] // 16  # Assuming patch size of 16
    #
    #     if old_grid_size == new_grid_size:
    #         return  # No interpolation needed
    #
    #     # Separate class token and position embeddings
    #     class_token = pos_embed[:, :1, :]
    #     pos_tokens = pos_embed[:, 1:, :]
    #
    #     # Reshape and interpolate
    #     pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, -1)
    #     pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # [1, embed_dim, H, W]
    #
    #     pos_tokens = F.interpolate(
    #         pos_tokens,
    #         size=(new_grid_size, new_grid_size),
    #         mode='bicubic',
    #         align_corners=False
    #     )
    #
    #     pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # Back to [1, N, embed_dim]
    #
    #     # Concatenate class token and interpolated position embeddings
    #     new_pos_embed = torch.cat([class_token, pos_tokens], dim=1)
    #     model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

    def _replace_classifier(
        self,
        model: nn.Module,
        model_name: str,
        num_classes: int
    ) -> nn.Module:
        """Replace classifier layer for custom number of classes."""
        # Get classifier path from registry
        model_cfg = self.registry.get(model_name)
        if model_cfg is None:
            classifier_path = self._auto_detect_classifier(model)
            print(f"⚠ Auto-detected classifier: {classifier_path}")
        else:
            classifier_path = model_cfg.classifier_path

        # Navigate to the layer
        parts = classifier_path.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        old_classifier = getattr(parent, attr_name) if not parts[-1].isdigit() else parent[int(parts[-1])]

        # Determine input features
        if isinstance(old_classifier, nn.Linear):
            in_features = old_classifier.in_features
        elif isinstance(old_classifier, nn.Sequential):
            # Find the last Linear layer
            for layer in reversed(old_classifier):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        else:
            raise ValueError(f"Unsupported classifier type: {type(old_classifier)}")

        # Create new classifier
        new_classifier = nn.Linear(in_features, num_classes)

        # Replace classifier
        if not parts[-1].isdigit():
            setattr(parent, attr_name, new_classifier)
        else:
            parent[int(parts[-1])] = new_classifier

        print(f"✓ Replaced classifier: {in_features} → {num_classes} classes")
        return model

    def _auto_detect_first_conv(self, model: nn.Module) -> str:
        """Auto-detect first convolutional layer."""
        common_paths = ['conv1', 'features.0.0', 'conv_proj', 'stem.0']
        for path in common_paths:
            try:
                parts = path.split('.')
                obj = model
                for part in parts:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                if isinstance(obj, nn.Conv2d):
                    return path
            except:
                continue
        raise ValueError("Could not auto-detect first conv layer. Please register the model.")

    def _auto_detect_classifier(self, model: nn.Module) -> str:
        """Auto-detect classifier layer."""
        common_paths = ['fc', 'classifier', 'head', 'heads.head', 'classifier.1']
        for path in common_paths:
            try:
                parts = path.split('.')
                obj = model
                for part in parts:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                if isinstance(obj, (nn.Linear, nn.Sequential)):
                    return path
            except:
                continue
        raise ValueError("Could not auto-detect classifier layer. Please register the model.")

    def _create_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer with smart defaults."""
        training_config = config.get('training', {})
        model_config = config['model']

        # Get model architecture family
        model_cfg = self.registry.get(model_config['name'])
        arch_family = model_cfg.architecture_family if model_cfg else 'cnn'

        # Smart defaults based on architecture
        if arch_family == 'cnn':
            default_optimizer = 'sgd'
            default_lr = 0.01
            default_momentum = 0.9
        else:  # transformer
            default_optimizer = 'adamw'
            default_lr = 3e-4
            default_momentum = None

        # Override with config if provided
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

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with smart defaults."""
        training_config = config.get('training', {})
        model_config = config['model']

        # Get model architecture family
        model_cfg = self.registry.get(model_config['name'])
        arch_family = model_cfg.architecture_family if model_cfg else 'cnn'

        # Smart defaults
        if arch_family == 'cnn':
            default_scheduler = 'step'
        else:
            default_scheduler = 'cosine'

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
# DATASET BUILDER
# ============================================================================

class DatasetBuilder:
    """
    Builds dataloaders from configuration or accepts pre-built loaders.
    Supports: torchvision datasets, custom datasets, ImageFolder.
    """

    @staticmethod
    def build(
        dataset_config: Union[Dict[str, Any], DataLoader],
        split: str = 'train'
    ) -> DataLoader:
        """
        Build dataloader from config or return pre-built loader.

        Args:
            dataset_config: Either a config dict or a DataLoader
            split: 'train' or 'val'
        """
        # Option A: Pre-built DataLoader
        if isinstance(dataset_config, DataLoader):
            return dataset_config

        # Option B: Build from config
        dataset_type = dataset_config['type']

        if dataset_type == 'torchvision':
            return DatasetBuilder._build_torchvision_dataset(dataset_config, split)
        elif dataset_type == 'custom':
            return DatasetBuilder._build_custom_dataset(dataset_config, split)
        elif dataset_type == 'folder':
            return DatasetBuilder._build_folder_dataset(dataset_config, split)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def _build_torchvision_dataset(config: Dict[str, Any], split: str) -> DataLoader:
        """Build torchvision dataset (CIFAR, ImageNet, etc.)."""
        dataset_name = config['name']
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)

        # Get dataset class
        dataset_class = getattr(datasets, dataset_name)

        # Define transforms
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4) if 'CIFAR' in dataset_name else transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256) if 'CIFAR' not in dataset_name else transforms.Resize(32),
                transforms.CenterCrop(224) if 'CIFAR' not in dataset_name else transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Create dataset
        is_train = (split == 'train')
        dataset = dataset_class(
            root=config.get('root', './data'),
            train=is_train,
            download=True,
            transform=transform
        )

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"✓ Built {split} torchvision dataset: {len(dataset)} samples")
        return loader

    @staticmethod
    def _build_custom_dataset(config: Dict[str, Any], split: str) -> DataLoader:
        """Build custom dataset using ClassifierDataset."""
        # Import your existing ClassifierDataset with correct path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        try:
            from resnet101_classifier.dataset import ClassifierDataset
        except ImportError as e:
            raise ImportError(
                f"Could not import ClassifierDataset. Make sure dataset.py is in the correct location. Error: {e}"
            )

        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)

        # Extract ClassifierDataset-specific parameters
        image_dir = config['path']
        channels = config.get('channels', 840)
        c_step = config.get('c_step', 10)
        state = config.get('state', 'before')
        label_type = config.get('label_type', 'ABCD')

        # Set transform based on split
        use_transform = (split == 'train')

        print(f"\nBuilding {split} dataset:")
        print(f"  - Image dir: {image_dir}")
        print(f"  - Channels: {channels} (step: {c_step})")
        print(f"  - State: {state}")
        print(f"  - Label type: {label_type}")
        print(f"  - Transform: {use_transform}")

        # Create dataset
        dataset = ClassifierDataset(
            image_dir=image_dir,
            channels=channels,
            c_step=c_step,
            transform=use_transform,
            state=state,
            label_type=label_type
        )

        # Verify dataset loaded correctly
        if dataset.images is None or len(dataset.images) == 0:
            raise ValueError(f"No images found in {image_dir} with state '{state}'")

        dataset_size = len(dataset.images)

        # Split dataset 80/20
        train_size = int(0.8 * dataset_size)

        # Since ClassifierDataset already shuffles images in __init__,
        # we can split sequentially
        if split == 'train':
            indices = list(range(train_size))
        else:  # val
            indices = list(range(train_size, dataset_size))

        # Create subset
        subset = Subset(dataset, indices)

        # Create dataloader
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        print(f"✓ Built {split} dataloader: {len(subset)}/{dataset_size} samples, batch_size={batch_size}\n")

        return loader

    @staticmethod
    def _build_folder_dataset(config: Dict[str, Any], split: str) -> DataLoader:
        """Build dataset from ImageFolder structure."""
        data_path = config['path']
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)

        # Define transforms
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Create dataset
        dataset_path = os.path.join(data_path, split)
        dataset = datasets.ImageFolder(dataset_path, transform=transform)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"✓ Built {split} dataloader from ImageFolder: {len(dataset)} samples")
        return loader


# ============================================================================
# GENERAL TRAINER
# ============================================================================

class GeneralTrainer:
    """
    Generic trainer for any classification model.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Checkpoint saving/loading
    - Metrics tracking + visualization
    - Early stopping
    - Multi-GPU support
    """

    def __init__(
        self,
        device: str = 'cuda',
        use_amp: bool = True,
        grad_accumulation_steps: int = 1,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10
    ):
        self.device = device
        self.use_amp = use_amp
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train(
        self,
        model_dict: Dict[str, Any],
        train_data: Union[DataLoader, Dict[str, Any]],
        val_data: Union[DataLoader, Dict[str, Any]],
        config: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Train model.

        Returns:
            dict: Training history
        """
        # Extract components
        model = model_dict['model']
        optimizer = model_dict['optimizer']
        scheduler = model_dict['scheduler']
        criterion = model_dict['criterion']

        # Build dataloaders if needed
        if not isinstance(train_data, DataLoader):
            train_loader = DatasetBuilder.build(train_data, split='train')
        else:
            train_loader = train_data

        if not isinstance(val_data, DataLoader):
            val_loader = DatasetBuilder.build(val_data, split='val')
        else:
            val_loader = val_data

        # Setup model
        model = self._setup_model(model)
        criterion = criterion.to(self.device)

        # Training parameters
        num_epochs = config['training']['num_epochs']

        # Training loop
        print(f"\n{'='*60}")
        print(f"Starting Training: {config['name']}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Use AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch, num_epochs
            )

            # Validate
            val_loss, val_acc, y_true, y_pred = self._validate_epoch(
                model, val_loader, criterion, epoch, num_epochs
            )

            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Save checkpoint if best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, output_dir, is_best=True
                )
                print(f"✓ New best model! Val Acc: {val_acc:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, output_dir, is_best=False
                )

        # Generate plots
        self._plot_training_curves(output_dir)
        self._plot_confusion_matrix(y_true, y_pred, output_dir)

        # Save classification report
        report = classification_report(y_true, y_pred)
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n✓ Training completed!")
        print(f"✓ Best val accuracy: {self.best_val_acc:.4f}")
        print(f"✓ Outputs saved to: {output_dir}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
        }

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training (device, multi-GPU)."""
        model = model.to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        return model

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        num_epochs: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.grad_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * self.grad_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': acc})

        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
        num_epochs: int
    ) -> Tuple[float, float, List, List]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store for confusion matrix
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Update progress bar
                acc = 100. * correct / total
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': acc})

        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_targets, all_predictions

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        output_dir: str,
        is_best: bool = False
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }

        if is_best:
            path = os.path.join(output_dir, 'best_model.pth')
        else:
            path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)

    def _load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def _plot_training_curves(self, output_dir: str):
        """Plot loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
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

    def _plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        output_dir: str
    ):
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


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """
    Orchestrates multiple experiments with error handling and resumption.
    """

    def __init__(self, base_output_dir: str = 'runs'):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        self.results = {}

    def run_experiments(self, configs: List[Dict[str, Any]]):
        """Run all experiments sequentially."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RUNNER")
        print(f"{'='*60}")
        print(f"Total experiments: {len(configs)}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"{'='*60}\n")

        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running: {config['name']}")
            success = self._run_single_experiment(config)
            self.results[config['name']] = 'SUCCESS' if success else 'FAILED'

        # Generate summary
        self._generate_summary_report()

    def _run_single_experiment(self, config: Dict[str, Any]) -> bool:
        """Run a single experiment with error handling."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{config['name']}_{timestamp}"
        output_dir = os.path.join(self.base_output_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save config
            config_path = os.path.join(output_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Build model
            builder = GenericModelBuilder()
            model_dict = builder.build(config)

            # Build dataloaders
            train_data = config['dataset']
            val_data = config['dataset']  # Same config, different split

            # Create trainer
            training_config = config['training']
            trainer = GeneralTrainer(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_amp=training_config.get('use_amp', True),
                grad_accumulation_steps=training_config.get('grad_accumulation_steps', 1),
                grad_clip=training_config.get('grad_clip', 1.0),
                early_stopping_patience=training_config.get('early_stopping_patience', 10)
            )

            # Train
            history = trainer.train(
                model_dict=model_dict,
                train_data=train_data,
                val_data=val_data,
                config=config,
                output_dir=output_dir
            )

            # Save history
            history_path = os.path.join(output_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

            print(f"✓ Experiment completed successfully: {config['name']}")
            return True

        except Exception as e:
            print(f"✗ Experiment failed: {config['name']}")
            print(f"Error: {str(e)}")
            trace = traceback.format_exc()
            self._save_error(config, e, trace, output_dir)
            return False

    def _save_error(
        self,
        config: Dict[str, Any],
        exception: Exception,
        trace: str,
        output_dir: str
    ):
        """Save error details to file."""
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

        print(f"✗ Error details saved to {error_path}")

    def _generate_summary_report(self):
        """Generate final summary of all experiments."""
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

    def resume_failed_experiments(self):
        """Detect and retry failed experiments."""
        print("\nSearching for failed experiments to resume...")

        failed_configs = []
        for exp_dir in os.listdir(self.base_output_dir):
            full_path = os.path.join(self.base_output_dir, exp_dir)
            if os.path.isdir(full_path):
                error_file = os.path.join(full_path, 'errors.txt')
                config_file = os.path.join(full_path, 'config.yaml')

                if os.path.exists(error_file) and os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    failed_configs.append(config)
                    print(f"  Found failed: {config['name']}")

        if not failed_configs:
            print("No failed experiments found.")
            return

        print(f"\nRetrying {len(failed_configs)} failed experiments...")
        self.run_experiments(failed_configs)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point - define your experiments here."""

    # Example experiments configuration
    EXPERIMENTS = [
        # Your existing multi-spectral experiment
        {
            'name': 'exp1_hyperspectral_640x640_4classes',
            'model': {
                'source': 'torchvision',
                'name': 'resnet101',
                'pretrained': True,
                'input_channels': 84,           # Your 84 channels (840/10)
                'input_size': (640, 640),       # Your spatial size
                'channel_adaptation': 'repeat',
                'num_classes': 4,
            },
            'dataset': {
                'type': 'custom',
                'path': '/home/ARO.local/tahor/PycharmProjects/data/pair_data',
                'channels': 840,
                'c_step': 10,
                'state': 'before',
                'label_type': 'ABCD',
                'batch_size': 10,
                'num_workers': 4,
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
            'hardware': {
                'multi_gpu': False,
                'gpu_ids': [0],
            },
        },

        {
            'name': 'exp2_hyperspectral_640x640_4classes',
            'model': {
                'source': 'torchvision',
                'name': 'vit_b_16',
                'pretrained': True,
                'input_channels': 84,  # Your 84 channels (840/10)
                'input_size': (640, 640),  # Your spatial size
                'channel_adaptation': 'repeat',
                'num_classes': 4,
            },
            'dataset': {
                'type': 'custom',
                'path': '/home/ARO.local/tahor/PycharmProjects/data/pair_data',
                'channels': 840,
                'c_step': 10,
                'state': 'before',
                'label_type': 'ABCD',
                'batch_size': 10,
                'num_workers': 4,
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
            'hardware': {
                'multi_gpu': False,
                'gpu_ids': [0],
            },
        },

        {
            'name': 'exp3_hyperspectral_640x640_4classes',
            'model': {
                'source': 'torchvision',
                'name': 'efficientnet_b0',
                'pretrained': True,
                'input_channels': 84,  # Your 84 channels (840/10)
                'input_size': (640, 640),  # Your spatial size
                'channel_adaptation': 'repeat',
                'num_classes': 4,
            },
            'dataset': {
                'type': 'custom',
                'path': '/home/ARO.local/tahor/PycharmProjects/data/pair_data',
                'channels': 840,
                'c_step': 10,
                'state': 'before',
                'label_type': 'ABCD',
                'batch_size': 10,
                'num_workers': 4,
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
            'hardware': {
                'multi_gpu': False,
                'gpu_ids': [0],
            },
        },
    ]

    # Run experiments
    runner = ExperimentRunner(base_output_dir='runs')
    runner.run_experiments(EXPERIMENTS)

    # Optional: Resume failed experiments
    # Uncomment the line below to retry failed experiments
    # runner.resume_failed_experiments()


if __name__ == '__main__':
    main()