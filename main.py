import torch
import argparse
from train import Config, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # General Settings
    parser.add_argument('--exp', type=str, default='default', help='Experiment name')
    parser.add_argument('--root', type=str, default='', help='Root directory for data')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (required)')
    parser.add_argument('--printing', type=bool, default=True, help='Enable printing/logging')
    parser.add_argument('--overwrite', type=bool, default=True, help='Overwrite existing logs')

    # Data and Preprocessing
    parser.add_argument('--split_ratio', type=str, default='0.8,0.2', help='Train/val split ratio')
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

    # Cross-Validation
    parser.add_argument('--cross_validation', type=bool, default=False, help='Enable cross-validation')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits for cross-validation')

    return parser.parse_args()

def main():
    args = parse_args()

    # Process split_ratio and loss_weights if provided
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    loss_weights = None
    if args.loss_weights:
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32) 

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
        'loss_weights': loss_weights
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


'''
python main.py --exp alexnet --dataset_name "kauhc" --model_name alexnet --epochs 100
python main.py --exp densenet121 --dataset_name "kauhc" --model_name densenet121 --epochs 100
python main.py --exp densenet201 --dataset_name "kauhc" --model_name densenet201 --epochs 100
python main.py --exp efficientnet_b0 --dataset_name "kauhc" --model_name efficientnet_b0 --epochs 100
python main.py --exp efficientnet_b7 --dataset_name "kauhc" --model_name efficientnet_b7 --epochs 100
python main.py --exp googlenet --dataset_name "kauhc" --model_name googlenet --epochs 100
python main.py --exp inception_v3 --dataset_name "kauhc" --model_name inception_v3 --epochs 100
python main.py --exp resnet50 --dataset_name "kauhc" --model_name resnet50 --epochs 100
python main.py --exp resnet152 --dataset_name "kauhc" --model_name resnet152 --epochs 100
python main.py --exp vgg16 --dataset_name "kauhc" --model_name vgg16 --epochs 100
python main.py --exp vgg19 --dataset_name "kauhc" --model_name vgg19 --epochs 100

python main.py --exp logistic_regression --dataset_name "kauhc" --model_name logistic_regression
python main.py --exp decision_tree --dataset_name "kauhc" --model_name decision_tree
python main.py --exp random_forest --dataset_name "kauhc" --model_name random_forest
python main.py --exp svm --dataset_name "kauhc" --model_name svm
python main.py --exp knn --dataset_name "kauhc" --model_name knn
python main.py --exp naive_bayes --dataset_name "kauhc" --model_name naive_bayes
python main.py --exp gbm --dataset_name "kauhc" --model_name gbm
python main.py --exp adaboost --dataset_name "kauhc" --model_name adaboost
python main.py --exp lda --dataset_name "kauhc" --model_name lda
python main.py --exp qda --dataset_name "kauhc" --model_name qda
python main.py --exp mlp --dataset_name "kauhc" --model_name mlp

'''