# Vision Classification Framework

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)
![Jupyter](https://img.shields.io/badge/jupyter-lab-orange.svg)

A comprehensive deep learning and machine learning framework for image classification tasks with support for multiple pretrained architectures, attention mechanisms, and traditional ML models. Built with PyTorch and designed for flexibility and ease of use.

## Overview

This framework provides a unified interface for training and evaluating both deep learning and traditional machine learning models on image classification tasks. It supports:

- **60+ pretrained deep learning architectures** from torchvision (ResNet, EfficientNet, ViT, MobileNet, etc.)
- **15+ attention mechanisms** (SE, CBAM, BAM, Coordinate Attention, etc.)
- **11 traditional ML models** (Logistic Regression, Random Forest, SVM, etc.)
- **Cross-validation support** for robust model evaluation
- **Comprehensive logging and visualization** with confusion matrices and performance metrics
- **Automated dataset handling** with support for compressed, uncompressed, and pre-split datasets
- **GPU acceleration** with CUDA support
- **Docker support** for reproducible environments

## Features

### Deep Learning Models
- **AlexNet:** alexnet
- **VGG:** vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
- **ResNet:** resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
- **ResNeXt:** resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
- **DenseNet:** densenet121, densenet161, densenet169, densenet201
- **EfficientNet:** efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
- **MobileNet:** mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **Vision Transformers:** vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
- **Swin Transformers:** swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
- **ConvNeXt:** convnext_tiny, convnext_small, convnext_base, convnext_large
- **RegNet:** regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_y_128gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, regnet_x_8gf, regnet_x_16gf, regnet_x_32gf
- **ShuffleNet:** shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
- **SqueezeNet:** squeezenet1_0, squeezenet1_1
- **MNASNet:** mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
- **MaxViT:** maxvit_t
- **GoogleNet:** googlenet
- **Inception:** inception_v3

### Attention Mechanisms
- **SE Layer:** Squeeze-and-Excitation Networks
- **CBAM:** Convolutional Block Attention Module
- **BAM:** Bottleneck Attention Module
- **Double Attention:** Double Attention Networks
- **SRM:** Style Recalibration Module
- **GC Module:** Global Context Module (GCNet)
- **SK Layer:** Selective Kernel Networks
- **LCT:** Linear Context Transform Block
- **GCT:** Gated Channel Transformation / Gaussian Context Transformer
- **ECA:** Efficient Channel Attention
- **Triplet Attention:** Rotate to Attend - Convolutional Triplet Attention Module
- **Coordinate Attention:** Coordinate Attention for Efficient Mobile Network Design
- **SimAM:** Simple Parameter-Free Attention Module
- **PAM:** Position Attention Module (from Dual Attention Network)
- **CAM:** Channel Attention Module (from Dual Attention Network)

### Traditional ML Models
- **Logistic Regression:** Linear classification with logistic function
- **Decision Tree:** Tree-based classification
- **Random Forest:** Ensemble of decision trees
- **SVM:** Support Vector Machine with RBF kernel
- **KNN:** K-Nearest Neighbors classifier
- **Naive Bayes:** Gaussian Naive Bayes classifier
- **GBM:** Gradient Boosting Machine
- **AdaBoost:** Adaptive Boosting classifier
- **LDA:** Linear Discriminant Analysis
- **QDA:** Quadratic Discriminant Analysis
- **MLP:** Multi-Layer Perceptron (Neural Network)

### Training Features
- **Learning Rate Scheduling:** StepLR, ReduceLROnPlateau, CosineAnnealingLR
- **Early Stopping:** Patience-based training termination to prevent overfitting
- **Training Visualization:** Automatic loss/accuracy curves and metrics plots
- **Per-Class Metrics:** Detailed precision, recall, F1 scores per class with CSV export
- **Model Summary:** Automatic parameter counting and size estimation
- **Data Augmentation:** Flip, rotation, resize transformations
- **Custom Loss Weights:** Support for imbalanced datasets
- **Automatic Checkpointing:** Save best models automatically
- **Cross-Validation:** Stratified K-fold support for robust evaluation
- **Comprehensive Metrics:** Accuracy, precision, recall, F1, confusion matrices
- **Production-Ready Inference API:** Easy deployment with batch prediction support

## Installation

### Using Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build -t vision-cls:latest .
```

2. **Run with Docker Compose:**
```bash
# Edit docker-compose.yml to set your parameters
docker-compose up
```

3. **Or run directly:**
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log vision-cls:latest \
  python3 main.py --dataset_name your_dataset --model_name resnet50 --epochs 50
```

### Manual Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd vision-cls-master
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Handling

This framework automates the dataset preparation process with flexible input options:

### Option 1: Compressed Dataset
- Place your zipped dataset file in `./data/compressed/`
- The zip file should contain a folder with class subfolders
- Use the same name (without .zip) for `--dataset_name` when running the script
- Example: If your file is `./data/compressed/my_dataset.zip`, use `--dataset_name my_dataset`

### Option 2: Uncompressed Dataset (Not Split)
- Place your uncompressed dataset folder in `./data/uncompressed/`
- Organize images into class subfolders: `./data/uncompressed/my_dataset/class1/`, `class2/`, etc.
- Use the folder name for `--dataset_name` when running the script
- The script will automatically split it into train/val/test sets

### Option 3: Pre-Split Dataset
- If your dataset is already split, place it directly in `./data/splitted/`
- Structure should be: `./data/splitted/my_dataset/train/`, `val/`, `test/`
- Each split folder should contain class subfolders
- The script will detect and use it without additional processing

### Dataset Structure Requirements

**For compressed or uncompressed datasets:**
```
your_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

**For pre-split datasets:**
```
your_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── class1/
│   └── class2/
└── test/                  # Optional
    ├── class1/
    └── class2/
```

### Split Ratios
- Default split ratio: `0.8:0.1:0.1` (train:val:test)
- For 2-way split use: `--split_ratio 0.8,0.2` (train:val)
- For 3-way split use: `--split_ratio 0.7,0.15,0.15` (train:val:test)
- The script automatically handles unzipping and splitting based on the split ratio

### Complete Directory Structure
```
vision-cls-master/
├── data/
│   ├── compressed/           # Place .zip files here
│   │   └── my_dataset.zip
│   ├── uncompressed/         # Place uncompressed datasets here
│   │   └── my_dataset/
│   │       ├── class1/
│   │       └── class2/
│   └── splitted/             # Auto-generated or pre-split datasets
│       └── my_dataset/
│           ├── train/
│           ├── val/
│           └── test/
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py              # Main entry point
│   ├── train.py             # Training logic
│   ├── datasets.py          # Dataset handling
│   ├── modules.py           # Model architectures
│   └── utils.py             # Utility functions
├── notebooks/                # Jupyter notebooks
│   ├── 01_getting_started.ipynb
│   ├── 02_attention_mechanisms.ipynb
│   └── 03_cross_validation.ipynb
├── log/                      # Training logs and outputs
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
├── run.sh                    # Start script (with Jupyter)
├── stop.sh                   # Stop script
├── README.md                 # This file
└── ISSUES_AND_LIMITATIONS.md # Known issues and limitations
```

## Usage

### Command Line Interface

#### Quick Start with Docker
```bash
# Start the container with Jupyter Lab
./run.sh

# Access Jupyter Lab at http://localhost:8888
# Or enter container bash:
docker exec -it vision-cls-container /bin/bash

# Inside container, run training:
python3 src/main.py --dataset_name your_dataset --model_name resnet50 --epochs 100

# Stop the container:
./stop.sh
```

#### Basic Training (Direct)
```bash
python src/main.py --dataset_name your_dataset --model_name resnet50 --epochs 100
```

#### Training with Attention Mechanism
```bash
python main.py \
  --dataset_name your_dataset \
  --model_name mobilenet_v2 \
  --attention_name cbam \
  --attention_index 4 \
  --epochs 100
```

#### Cross-Validation Training
```bash
python main.py \
  --dataset_name your_dataset \
  --model_name efficientnet_b0 \
  --cross_validation True \
  --num_splits 5 \
  --epochs 50
```

#### Training Traditional ML Model
```bash
python main.py \
  --dataset_name your_dataset \
  --model_name random_forest
```

#### Advanced Configuration
```bash
python main.py \
  --dataset_name your_dataset \
  --model_name resnet50 \
  --attention_name se_layer \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epochs 100 \
  --image_size 224 \
  --split_ratio 0.7,0.15,0.15 \
  --aug True \
  --loss_weights 1.0,2.0,1.5 \
  --lr_scheduler plateau \
  --early_stopping 15
```

#### Learning Rate Scheduling
```bash
# StepLR: Decay LR every 30 epochs by 0.1
python src/main.py \
  --dataset_name your_dataset \
  --model_name resnet50 \
  --lr_scheduler step \
  --epochs 100

# ReduceLROnPlateau: Reduce LR when validation loss plateaus
python src/main.py \
  --dataset_name your_dataset \
  --model_name resnet50 \
  --lr_scheduler plateau \
  --epochs 100

# CosineAnnealingLR: Cosine annealing schedule
python src/main.py \
  --dataset_name your_dataset \
  --model_name efficientnet_b0 \
  --lr_scheduler cosine \
  --epochs 100
```

#### Early Stopping
```bash
# Stop training if no improvement for 10 epochs
python src/main.py \
  --dataset_name your_dataset \
  --model_name resnet50 \
  --early_stopping 10 \
  --epochs 100

# Combined with LR scheduling
python src/main.py \
  --dataset_name your_dataset \
  --model_name resnet50 \
  --lr_scheduler plateau \
  --early_stopping 15 \
  --epochs 100
```

### Using in Google Colab

#### Setup in Colab
```python
# Clone the repository
!git clone git@github.com:0aub/vision-cls-master.git

# Change to repository directory
%cd vision-cls-master

# Install dependencies
!pip install -q -r requirements.txt

# Now you can import and use
import sys
sys.path.append('/content/vision-cls-master')

from src.train import Config, Trainer

# Configure and train
config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'resnet50',
    'epochs': 20,
    'batch_size': 16
}

config = Config(config_dict)
trainer = Trainer(config)
trainer.run()
```

**Note:** Make sure to upload your dataset to Colab or mount Google Drive:
```python
# Mount Google Drive (if dataset is on Drive)
from google.colab import drive
drive.mount('/content/drive')

# Then set root path
config_dict = {
    'root': '/content/drive/MyDrive/',
    'dataset_name': 'your_dataset',
    # ... other parameters
}
```

### Using as Python Module

#### Basic Usage
```python
from src.train import Config, Trainer

# Configure training parameters
config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'resnet50',
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.0001,
    'image_size': 256,
    'lr_scheduler': 'plateau',  # Optional: step, plateau, cosine
    'early_stopping': 10,        # Optional: patience for early stopping
    'train': True,
    'eval': True,
    'save': True
}

# Initialize and run training
config = Config(config_dict)
trainer = Trainer(config)
trainer.run()
```

#### Inference API
```python
from src.inference import Predictor

# Initialize predictor
predictor = Predictor(
    model_path='log/experiment/best.pth',
    model_name='resnet50',
    classes=['cat', 'dog', 'bird']
)

# Single image prediction
result = predictor.predict_image('test.jpg')
print(f"Predicted: {result['top_prediction']} ({result['top_probability']:.2%})")

# Batch inference
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for r in results:
    print(f"{r['image_path']}: {r['predicted_class']} ({r['confidence']})")

# Predict all images in a folder
results = predictor.predict_from_folder('test_images/')

# Get probabilities for all classes
probs = predictor.get_class_probabilities('test.jpg')
for class_name, prob in probs.items():
    print(f"{class_name}: {prob:.4f}")
```

#### Quick Single Prediction
```python
from src.inference import predict_single

# Quick prediction without creating a Predictor object
predicted_class = predict_single(
    model_path='log/experiment/best.pth',
    image_path='test.jpg',
    model_name='resnet50',
    classes=['cat', 'dog']
)
print(f"Predicted: {predicted_class}")
```

#### With Attention Mechanism
```python
from train import Config, Trainer

config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'mobilenet_v2',
    'attention_name': 'cbam',
    'attention_index': 4,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0001
}

config = Config(config_dict)
trainer = Trainer(config)
trainer.run()
```

#### Cross-Validation
```python
from train import Config, Trainer

config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'efficientnet_b0',
    'cross_validation': True,
    'num_splits': 5,
    'epochs': 50,
    'batch_size': 16
}

config = Config(config_dict)
trainer = Trainer(config)
trainer.run()
```

#### Traditional ML Model
```python
from train import Config, Trainer

config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'random_forest',
    'train': True,
    'eval': True,
    'save': True
}

config = Config(config_dict)
trainer = Trainer(config)
trainer.run_ml()
```

#### Custom Training Loop
```python
import torch
from train import Config, Trainer
from datasets import ImageDataset
from modules import pretrained_network

# Custom configuration
config_dict = {
    'dataset_name': 'your_dataset',
    'model_name': 'resnet50',
    'batch_size': 16,
    'image_size': 224,
    'learning_rate': 0.0001,
    'epochs': 100,
    'split_ratio': (0.8, 0.1, 0.1)
}

config = Config(config_dict)
trainer = Trainer(config)

# Access components directly
model = trainer.model
optimizer = trainer.optimizer
train_loader = trainer.dataset.dataloaders['train']
val_loader = trainer.dataset.dataloaders['val']

# Custom training logic
best_model_wts, history = trainer.train(train_loader, val_loader)

# Evaluate
trainer.model.load_state_dict(best_model_wts)
trainer.evaluate('Test', trainer.dataset.dataloaders['test'])
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exp` | str | 'default' | Experiment name for logging |
| `root` | str | '' | Root directory for data |
| `dataset_name` | str | **required** | Name of the dataset |
| `model_name` | str | 'mobilenet_v2' | Model architecture name |
| `attention_name` | str | None | Attention mechanism name |
| `attention_index` | int | 4 | Layer index to insert attention |
| `batch_size` | int | 16 | Batch size for training |
| `learning_rate` | float | 0.0001 | Learning rate |
| `epochs` | int | 100 | Number of training epochs |
| `image_size` | int | 256 | Image size (width/height) |
| `split_ratio` | str | '0.8,0.1,0.1' | Train/val/test split ratio |
| `aug` | bool | True | Enable data augmentation |
| `loss_weights` | str | None | Class weights for loss (comma-separated) |
| `cross_validation` | bool | False | Enable cross-validation |
| `num_splits` | int | 5 | Number of CV folds |
| `lr_scheduler` | str | None | LR scheduler: 'step', 'plateau', 'cosine' |
| `early_stopping` | int | None | Early stopping patience (epochs) |
| `train` | bool | True | Enable training |
| `eval` | bool | True | Enable evaluation |
| `save` | bool | True | Enable model saving |
| `printing` | bool | True | Enable logging output |
| `overwrite` | bool | True | Overwrite existing logs |

## Output and Logs

Training outputs are saved in the `log/` directory:

```
log/
└── experiment_name_YYYY-MM-DD_HH-MM-SS/
    ├── args.txt                      # Training configuration
    ├── log.txt                       # Training logs
    ├── history.csv                   # Training history (loss, accuracy, lr, etc.)
    ├── eval.txt                      # Evaluation results
    ├── best.pth                      # Best model weights (DL)
    ├── best.pkl                      # Best model weights (ML)
    ├── model_summary.txt             # Model parameters and size (NEW)
    ├── training_curves.png           # Loss/accuracy plots (NEW)
    ├── metrics_curves.png            # Precision/recall/F1 plots (NEW)
    ├── per_class_metrics_train.csv   # Per-class metrics for train set (NEW)
    ├── per_class_metrics_val.csv     # Per-class metrics for val set (NEW)
    ├── per_class_metrics_test.csv    # Per-class metrics for test set (NEW)
    ├── cm_train.png                  # Training confusion matrix
    ├── cm_val.png                    # Validation confusion matrix
    ├── cm_test.png                   # Test confusion matrix
    └── fold_*.pth/pkl                # Cross-validation fold models
```

### New Output Files Explained

- **model_summary.txt**: Contains total parameters, trainable parameters, and estimated model size in MB
- **training_curves.png**: Visualization of training/validation loss and accuracy over epochs
- **metrics_curves.png**: Visualization of precision, recall, and F1 score over epochs
- **per_class_metrics_*.csv**: Detailed per-class performance (precision, recall, F1, support) for each split
- **history.csv**: Now includes learning rate for each epoch (when using LR schedulers)

## Docker Commands

### Build Image
```bash
docker build -t vision-cls:latest .
```

### Run Training
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  vision-cls:latest \
  python3 main.py --dataset_name my_dataset --model_name resnet50 --epochs 50
```

### Run with Docker Compose
```bash
# Start training
docker-compose up

# Run in background
docker-compose up -d

# Stop training
docker-compose down

# View logs
docker-compose logs -f
```

### Interactive Shell
```bash
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  vision-cls:latest \
  /bin/bash
```

## Examples

### Example 1: Image Classification with ResNet50
```bash
python main.py \
  --dataset_name cifar10 \
  --model_name resnet50 \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 0.001
```

### Example 2: MobileNet with SE Attention
```bash
python main.py \
  --dataset_name imagenet_subset \
  --model_name mobilenet_v2 \
  --attention_name se_layer \
  --attention_index 4 \
  --batch_size 64 \
  --epochs 50
```

### Example 3: Cross-Validation with EfficientNet
```bash
python main.py \
  --dataset_name medical_images \
  --model_name efficientnet_b0 \
  --cross_validation True \
  --num_splits 10 \
  --epochs 100
```

### Example 4: Traditional ML with Random Forest
```bash
python main.py \
  --dataset_name small_dataset \
  --model_name random_forest
```

### Example 5: Production Inference
```python
from src.inference import Predictor

# Load trained model
predictor = Predictor(
    model_path='log/experiment_2024-11-27_12-00-00/best.pth',
    model_name='resnet50',
    classes=['cat', 'dog', 'bird', 'fish']
)

# Single image prediction with top-3 results
result = predictor.predict_image('test.jpg', top_k=3)
print(f"\nTop prediction: {result['top_prediction']}")
print(f"Confidence: {result['top_probability']:.2%}\n")

print("All predictions:")
for pred in result['predictions']:
    print(f"  {pred['class']}: {pred['confidence']}")

# Batch prediction
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_list, batch_size=32)

# Process all images in a folder
folder_results = predictor.predict_from_folder('validation_images/')
print(f"\nProcessed {len(folder_results)} images")
```

## Supported Models

### Deep Learning Models
- **AlexNet**: alexnet
- **VGG**: vgg11, vgg13, vgg16, vgg19 (+ bn variants)
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **ResNeXt**: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
- **DenseNet**: densenet121, densenet161, densenet169, densenet201
- **EfficientNet**: efficientnet_b0 to efficientnet_b7, efficientnet_v2_s/m/l
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **ViT**: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
- **Swin**: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base, convnext_large
- **And more**: RegNet, MNASNet, SqueezeNet, ShuffleNet, MaxViT, GoogleNet, Inception

### Attention Mechanisms
- `se_layer`: Squeeze-and-Excitation
- `cbam`: Convolutional Block Attention Module
- `bam`: Bottleneck Attention Module
- `double_attention`: Double Attention Networks
- `srm`: Style Recalibration Module
- `gc_module`: Global Context Module
- `sk_layer`: Selective Kernel
- `lct`: Linear Context Transform
- `gct`: Gated Channel Transformation
- `eca`: Efficient Channel Attention
- `triplet_attention`: Triplet Attention
- `coordinate_attention`: Coordinate Attention
- `simam`: Simple Parameter-Free Attention
- `pam`: Position Attention Module
- `cam`: Channel Attention Module

### Machine Learning Models
- `logistic_regression`: Logistic Regression
- `decision_tree`: Decision Tree
- `random_forest`: Random Forest
- `svm`: Support Vector Machine
- `knn`: K-Nearest Neighbors
- `naive_bayes`: Gaussian Naive Bayes
- `gbm`: Gradient Boosting Machine
- `adaboost`: AdaBoost
- `lda`: Linear Discriminant Analysis
- `qda`: Quadratic Discriminant Analysis
- `mlp`: Multi-Layer Perceptron

## Advanced Features

### Learning Rate Scheduling

Three built-in schedulers are available:

1. **StepLR**: Reduces learning rate by a factor (gamma=0.1) every 30 epochs
   ```bash
   python src/main.py --dataset_name data --model_name resnet50 --lr_scheduler step
   ```

2. **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
   ```bash
   python src/main.py --dataset_name data --model_name resnet50 --lr_scheduler plateau
   ```

3. **CosineAnnealingLR**: Cosine annealing schedule over the training period
   ```bash
   python src/main.py --dataset_name data --model_name resnet50 --lr_scheduler cosine
   ```

### Early Stopping

Prevent overfitting by stopping training when validation loss stops improving:

```bash
# Stop if no improvement for 15 epochs
python src/main.py --dataset_name data --model_name resnet50 --early_stopping 15
```

Combine with LR scheduling for optimal results:
```bash
python src/main.py --dataset_name data --lr_scheduler plateau --early_stopping 15
```

### Visualization and Analysis

The framework automatically generates:

- **Training Curves**: Loss and accuracy plots (`training_curves.png`)
- **Metrics Curves**: Precision, recall, F1 score plots (`metrics_curves.png`)
- **Per-Class Metrics**: Detailed CSV files for each class's performance
- **Confusion Matrices**: Visual representation of predictions vs actual labels
- **Model Summary**: Parameter count and memory usage estimation

### Inference API Features

The `Predictor` class provides production-ready inference:

```python
from src.inference import Predictor

predictor = Predictor('log/exp/best.pth', 'resnet50', ['cat', 'dog'])

# Single image
predictor.predict_image('test.jpg')

# Batch processing
predictor.predict_batch(['img1.jpg', 'img2.jpg'], batch_size=32)

# Folder processing
predictor.predict_from_folder('test_images/')

# PIL Image support
from PIL import Image
img = Image.open('test.jpg')
predictor.predict_with_PIL_image(img)

# Get all class probabilities
predictor.get_class_probabilities('test.jpg')
```

## GPU Support

The framework automatically detects and uses CUDA-enabled GPUs when available. For Docker:

```bash
# Ensure nvidia-docker is installed
docker run --gpus all vision-cls:latest python3 main.py --dataset_name your_dataset
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is available for educational and research purposes.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vision_cls_framework,
  title = {Vision Classification Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/vision-cls-master}
}
```

## Acknowledgments

- PyTorch and torchvision for pretrained models
- Attention mechanisms implementations inspired by various research papers
- scikit-learn for traditional ML models

## References

- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Attention Mechanisms](https://github.com/changzy00/pytorch-attention)
