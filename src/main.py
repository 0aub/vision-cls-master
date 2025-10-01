import torch
import argparse
from src.train import Config, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # General Settings
    parser.add_argument('--exp', type=str, default='default', help='Experiment name')
    parser.add_argument('--root', type=str, default='', help='Root directory for data')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (required)')
    parser.add_argument('--printing', type=bool, default=True, help='Enable printing/logging')
    parser.add_argument('--overwrite', type=bool, default=True, help='Overwrite existing logs')

    # Data and Preprocessing
    parser.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1', help='Train/val split ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--aug', type=bool, default=True, help='Enable data augmentation')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--loss_weights', type=str, default=None, help='Loss weights (comma-separated values)')

    # Model Configuration
    parser.add_argument('--model_name', type=str, default='mobilenet_v2', help='Model architecture name')
    parser.add_argument('--attention_name', type=str, default=None, help='Attention mechanism name')
    parser.add_argument('--attention_index', type=int, default=4, help='Attention layer index')

    # Training and Evaluation
    parser.add_argument('--train', type=bool, default=True, help='Enable training')
    parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation')
    parser.add_argument('--save', type=bool, default=True, help='Enable saving models')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'step', 'plateau', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--early_stopping', type=int, default=None, help='Early stopping patience (epochs)')

    # Cross-Validation
    parser.add_argument('--cross_validation', type=bool, default=False, help='Enable cross-validation')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits for cross-validation')

    return parser.parse_args()

def main():
    args = parse_args()

    # Validate parameters
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}")

    # Process split_ratio and loss_weights if provided
    try:
        split_ratio = tuple(map(float, args.split_ratio.split(',')))
        if len(split_ratio) not in [2, 3]:
            raise ValueError("split_ratio must have 2 or 3 values")
        if not all(0 < x < 1 for x in split_ratio):
            raise ValueError("split_ratio values must be between 0 and 1")
        if abs(sum(split_ratio) - 1.0) > 0.01:
            raise ValueError(f"split_ratio must sum to 1.0, got {sum(split_ratio)}")
    except Exception as e:
        raise ValueError(f"Invalid split_ratio '{args.split_ratio}': {e}")

    loss_weights = None
    if args.loss_weights:
        try:
            loss_weights = torch.tensor(list(map(float, args.loss_weights.split(','))), dtype=torch.float32)
            if torch.any(loss_weights < 0):
                raise ValueError("loss_weights must be non-negative")
        except Exception as e:
            raise ValueError(f"Invalid loss_weights '{args.loss_weights}': {e}") 

    # Build the configuration dictionary
    config_dict = {
        'exp': args.exp,
        'root': args.root,
        'dataset_name': args.dataset_name,
        'split_ratio': split_ratio,
        'cross_validation': args.cross_validation,
        'num_splits': args.num_splits,
        'model_name': args.model_name,
        'attention_name': args.attention_name,
        'attention_index': args.attention_index,
        'train': args.train,
        'eval': args.eval,
        'save': args.save,
        'overwrite': args.overwrite,
        'batch_size': args.batch_size,
        'aug': args.aug,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'printing': args.printing,
        'loss_weights': loss_weights,
        'lr_scheduler': args.lr_scheduler,
        'early_stopping': args.early_stopping
    }

    # Initialize Config and Trainer
    config = Config(config_dict)
    trainer = Trainer(config)
    
    # Start training
    if config.model_type == 'dl':
        trainer.run()
    else:
        trainer.run_ml()

if __name__ == "__main__":
    main()

