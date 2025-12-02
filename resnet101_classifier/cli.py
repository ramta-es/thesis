import argparse
import torch
import sys
import unittest
from classifier_V3 import main

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='classifier_tests.py')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet101 Classifier for Hyperspectral Images')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of output classes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--biased_class', type=int, default=None, help='Class index for biased accuracy (optional)')
    return parser.parse_args()

if __name__ == '__main__':
    run_tests()  # Run tests first
    args = parse_args()
    main(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        device=args.device,
        biased_class=args.biased_class
    )
