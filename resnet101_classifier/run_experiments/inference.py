"""
Inference script for fine-tuned classification models.

This script provides a command-line interface for running predictions on new datasets
using trained classification models. It supports multiple image formats, batch processing,
and various output options.

Usage Examples:
    # Basic inference
    python inference.py --checkpoint runs/exp2/best_model.pth --input_dir /data/images

    # With custom output and format
    python inference.py \
        --checkpoint runs/exp2/best_model.pth \
        --input_dir /data/test_images \
        --output_csv predictions.csv \
        --image_format tif

    # CPU-only inference with verbose output
    python inference.py \
        --checkpoint runs/model.pth \
        --input_dir /data/images \
        --device cpu \
        --verbose

Author: GitHub Copilot
Date: 2024
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Import your existing model builder
try:
    from run_experiments.trainer_scipt_V2 import GenericModelBuilder
except ImportError:
    print("ERROR: Could not import GenericModelBuilder from trainer_scipt_V2")
    print("Please ensure the script is in the correct directory structure.")
    sys.exit(1)


class InferenceEngine:
    """Handles model loading and prediction for classification tasks."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = False
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            verbose: Enable verbose logging
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.verbose = verbose

        # Load checkpoint and metadata
        if verbose:
            print(f"\n{'='*70}")
            print("LOADING MODEL CHECKPOINT")
            print(f"{'='*70}")

        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = self._load_config()

        # Build model
        self.model = self._build_model()
        self.model.eval()

        # Get label mapping
        self.label_mapping = self._get_label_mapping()
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        if verbose:
            print(f"{'='*70}\n")

    def _load_config(self) -> Dict:
        """Load experiment config from checkpoint directory."""
        config_path = self.checkpoint_path.parent / 'config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "Cannot auto-detect model architecture."
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if self.verbose:
            print(f"✓ Loaded config from: {config_path}")

        return config

    def _build_model(self) -> nn.Module:
        """Rebuild model architecture and load weights."""
        builder = GenericModelBuilder()

        # Build model using original config
        components = builder.build(self.config)
        model = components['model']

        # Load trained weights
        if 'model_state_dict' in self.checkpoint:
            state_dict = self.checkpoint['model_state_dict']
        else:
            # Handle old checkpoint format
            state_dict = self.checkpoint

        # Remove 'module.' prefix if model was trained with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(self.device)

        if self.verbose:
            print(f"✓ Loaded model: {self.config['model']['name']}")
            print(f"  • Input channels: {self.config['model']['input_channels']}")
            print(f"  • Input size: {self.config['model']['input_size']}")
            print(f"  • Num classes: {self.config['model']['num_classes']}")
            print(f"  • Device: {self.device}")

        return model

    def _get_label_mapping(self) -> Dict[str, int]:
        """Extract label mapping from config."""
        label_type = self.config['data'].get('label_type', 'ABCD')

        label_mappings = {
            'ABCD': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
            'AD': {'A': 0, 'D': 1},
            'AB-CD': {'A': 0, 'B': 0, 'C': 1, 'D': 1},
            'A-BCD': {'A': 0, 'B': 1, 'C': 1, 'D': 1}
        }

        mapping = label_mappings.get(label_type, label_mappings['ABCD'])

        if self.verbose:
            print(f"✓ Label mapping: {label_type}")
            for label, idx in sorted(mapping.items()):
                print(f"  • {label} → {idx}")

        return mapping

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file (.npy, .jpg, .png, etc.)

        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        # Load image based on format
        if image_path.suffix == '.npy':
            image = np.load(str(image_path)).astype(np.float32)
        else:
            # Handle standard image formats (jpg, png, tif)
            from PIL import Image
            image = np.array(Image.open(image_path)).astype(np.float32)

        # Apply channel selection from config
        channels = self.config['data']['channels']
        c_step = self.config['data']['c_step']

        if image.ndim == 3 and image.shape[2] >= channels:
            # Select channels with step
            selected_channels = [i for i in range(channels) if i % c_step == 0]
            image = image[:, :, selected_channels]

        # Normalize to [0, 1] range
        if image.max() > 1.0:
            image = image / 4096.0  # Assuming 12-bit hyperspectral data

        # Convert to tensor: [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        image = image.unsqueeze(0)

        return image

    @torch.no_grad()
    def predict(
        self,
        image_path: Path
    ) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (predicted_label, confidence, class_probabilities)
        """
        # Preprocess image
        image = self.preprocess_image(image_path).to(self.device)

        # Forward pass
        logits = self.model(image)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Get prediction
        predicted_class_idx = int(probabilities.argmax())
        predicted_label = self.reverse_label_mapping[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        return predicted_label, confidence, probabilities

    def predict_directory(
        self,
        input_dir: Path,
        output_csv: Path,
        image_format: str = 'npy',
        recursive: bool = True
    ):
        """
        Run inference on all images in a directory.

        Args:
            input_dir: Directory containing images
            output_csv: Path to output CSV file
            image_format: Image file extension (npy, jpg, png, tif)
            recursive: Search subdirectories recursively
        """
        # Find all image files
        if recursive:
            if image_format == 'npy':
                image_paths = list(input_dir.rglob('*.npy'))
            else:
                image_paths = list(input_dir.rglob(f'*.{image_format}'))
        else:
            if image_format == 'npy':
                image_paths = list(input_dir.glob('*.npy'))
            else:
                image_paths = list(input_dir.glob(f'*.{image_format}'))

        if not image_paths:
            raise ValueError(
                f"No {image_format} files found in {input_dir}" +
                (" (including subdirectories)" if recursive else "")
            )

        print(f"\n{'='*70}")
        print("RUNNING INFERENCE")
        print(f"{'='*70}")
        print(f"Model: {self.config['model']['name']}")
        print(f"Input directory: {input_dir}")
        print(f"Image format: .{image_format}")
        print(f"Recursive search: {'Yes' if recursive else 'No'}")
        print(f"Found images: {len(image_paths)}")
        print(f"Output: {output_csv}")
        print(f"{'='*70}\n")

        # Prepare CSV header
        class_labels = sorted(self.label_mapping.keys())
        csv_header = ['image_path', 'predicted_class', 'confidence'] + \
                     [f'prob_class_{label}' for label in class_labels]

        # Track statistics
        processed_count = 0
        failed_count = 0

        # Open CSV for writing
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

            # Process images one by one
            for image_path in tqdm(image_paths, desc="Predicting", unit="image"):
                try:
                    predicted_label, confidence, probabilities = self.predict(image_path)

                    # Write row to CSV
                    row = [
                        str(image_path),
                        predicted_label,
                        f'{confidence:.4f}'
                    ] + [f'{prob:.4f}' for prob in probabilities]

                    writer.writerow(row)
                    processed_count += 1

                except Exception as e:
                    failed_count += 1
                    if self.verbose:
                        warnings.warn(f"Failed to process {image_path}: {str(e)}")
                    continue

        # Print summary
        print(f"\n{'='*70}")
        print("INFERENCE COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Predictions saved to: {output_csv}")
        print(f"✓ Successfully processed: {processed_count}/{len(image_paths)} images")
        if failed_count > 0:
            print(f"⚠ Failed to process: {failed_count} images")
        print(f"{'='*70}\n")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        prog='inference.py',
        description='Run inference with fine-tuned classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with .npy files
  python inference.py \\
      --checkpoint runs/exp2_hyperspectral_640x640_4classes/best_model.pth \\
      --input_dir /path/to/new/data \\
      --output_csv predictions.csv

  # Inference with TIFF images (non-recursive)
  python inference.py \\
      --checkpoint runs/best_model.pth \\
      --input_dir /data/test_images \\
      --output_csv results.csv \\
      --image_format tif \\
      --no-recursive

  # CPU inference with verbose output
  python inference.py \\
      --checkpoint runs/model.pth \\
      --input_dir /data/images \\
      --device cpu \\
      --verbose

  # Quick test on single directory
  python inference.py \\
      --checkpoint runs/exp1/best_model.pth \\
      --input_dir /data/test \\
      --image_format png \\
      --no-recursive

For more information, see the project documentation.
        """
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to trained model checkpoint (.pth file)'
    )
    required.add_argument(
        '--input_dir',
        type=str,
        required=True,
        metavar='PATH',
        help='Directory containing images for inference'
    )

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output_csv',
        type=str,
        default='predictions.csv',
        metavar='PATH',
        help='Path to output CSV file (default: predictions.csv)'
    )

    # Input options
    input_group = parser.add_argument_group('input options')
    input_group.add_argument(
        '--image_format',
        type=str,
        default='npy',
        choices=['npy', 'jpg', 'jpeg', 'png', 'tif', 'tiff'],
        metavar='FORMAT',
        help='Image file format: npy, jpg, jpeg, png, tif, tiff (default: npy)'
    )
    input_group.add_argument(
        '--recursive',
        dest='recursive',
        action='store_true',
        default=True,
        help='Search subdirectories recursively (default: enabled)'
    )
    input_group.add_argument(
        '--no-recursive',
        dest='recursive',
        action='store_false',
        help='Disable recursive subdirectory search'
    )

    # Device options
    device_group = parser.add_argument_group('device options')
    device_group.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        metavar='DEVICE',
        help='Device to run inference on: cuda or cpu (default: auto-detect)'
    )

    # Display options
    display_group = parser.add_argument_group('display options')
    display_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    display_group.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress all non-essential output'
    )

    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments

    Raises:
        FileNotFoundError: If checkpoint or input directory doesn't exist
        ValueError: If arguments are invalid
    """
    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint must be a file: {checkpoint_path}")

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    # Validate output directory
    output_csv = Path(args.output_csv)
    if not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    # Conflicting options
    if args.verbose and args.quiet:
        raise ValueError("Cannot use both --verbose and --quiet")


def main():
    """Main entry point for inference script."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Suppress warnings if quiet mode
    if args.quiet:
        warnings.filterwarnings('ignore')

    # Print banner if not quiet
    if not args.quiet:
        print("\n" + "="*70)
        print("CLASSIFICATION MODEL INFERENCE")
        print("="*70)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Input: {args.input_dir}")
        print(f"Output: {args.output_csv}")
        print(f"Device: {args.device}")
        print("="*70)

    try:
        # Initialize inference engine
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            verbose=args.verbose and not args.quiet
        )

        # Run predictions
        engine.predict_directory(
            input_dir=Path(args.input_dir),
            output_csv=Path(args.output_csv),
            image_format=args.image_format,
            recursive=args.recursive
        )

    except Exception as e:
        print(f"\nERROR: Inference failed with exception:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()