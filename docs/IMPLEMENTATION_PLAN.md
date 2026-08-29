# Implementation Plan - Phase 2

**Status:** Ready to implement
**Date Created:** 2024-11-27
**Phase 1 Completed:** 13/13 features (6 bug fixes + 7 new features)
**Phase 2 Target:** 15 additional features (7 High Priority + 8 Medium Priority)

---

## Quick Reference - What's Done vs What's Next

### ✅ Phase 1 Complete (13 Features)
1. ✅ Loss weights parsing fix
2. ✅ ML CV model loading fix
3. ✅ DL CV optimizer reset fix
4. ✅ Inconsistent model saving fix
5. ✅ Dataset CV code duplication fix
6. ✅ ML CV data preparation fix
7. ✅ Error handling & validation
8. ✅ Training curves visualization
9. ✅ Per-class metrics
10. ✅ Learning rate scheduling (3 schedulers)
11. ✅ Early stopping
12. ✅ Inference/Prediction API
13. ✅ Model summary

### 🎯 Phase 2 - To Implement (15 Features)

#### **HIGH PRIORITY (7 features)**

**#9 - Resume/Checkpoint Capability**
- Save/load optimizer state, scheduler state, epoch counter, best metrics
- `--resume PATH` flag to continue from checkpoint
- Auto-save checkpoints every N epochs (`--checkpoint_freq`)
- Files: `src/train.py`, `src/main.py`

**#11 - ROC Curves, AUC, PR Curves**
- ROC curve plotting for binary/multi-class
- AUC calculation
- Precision-Recall curves
- Save as `roc_curve.png`, `pr_curve.png`, `auc_scores.txt`
- Files: `src/utils.py`, `src/train.py`

**#13 - Mixed Precision Training (AMP)**
- `torch.cuda.amp.GradScaler` + `autocast`
- `--mixed_precision` flag
- Automatic FP16 training on compatible GPUs
- Files: `src/train.py`, `src/main.py`

**#14 - Gradient Accumulation**
- `--accumulation_steps N` parameter
- Accumulate gradients over N batches
- Effective batch size = batch_size × accumulation_steps
- Files: `src/train.py`, `src/main.py`

**#15 - Fix CV Issues**
- Hold out test set from CV folds
- Save per-fold history (`fold_1_history.csv`, etc.)
- Ensemble predictions (voting/averaging)
- Best fold selection logic
- Final test set evaluation
- Files: `src/train.py`

**#21 - Multi-GPU Support**
- `torch.nn.DataParallel` for single-node
- `torch.nn.parallel.DistributedDataParallel` option
- `--gpus 0,1,2,3` flag
- Auto batch size scaling
- Files: `src/train.py`, `src/main.py`

**#23 - Unit and Integration Tests**
- Create `tests/` directory
- `test_datasets.py` - dataset loading tests
- `test_train.py` - training loop tests
- `test_inference.py` - inference API tests
- `test_utils.py` - utility function tests
- Use pytest framework
- Files: NEW - `tests/*.py`, `pytest.ini`

#### **MEDIUM PRIORITY (8 features)**

**#8 - Dataset Download Functionality**
- Implement `_download()` in datasets.py
- Support CIFAR10, CIFAR100, MNIST, ImageNet subset
- Auto-download with progress bars (tqdm)
- Hash verification
- `--auto_download` flag
- Files: `src/datasets.py`, `src/main.py`

**#16 - Fix Dataset Handling Limitations**
- Support absolute paths (not just relative)
- Support symlinks
- Auto-call `_clean()` for corrupted images
- Support grayscale (1-channel) images
- Configurable normalization statistics
- `--normalize_mean` and `--normalize_std` flags
- Dataset validation before splitting
- Files: `src/datasets.py`, `src/main.py`

**#17 - YAML/JSON Config File Support**
- NEW FILE: `src/config.py` - config loader
- `--config config.yaml` CLI argument
- Override config with CLI args (CLI > config file > defaults)
- Save final config to `log/experiment/config_used.yaml`
- Create `config_templates/` with examples
- Schema validation
- Files: NEW - `src/config.py`, `config_templates/*.yaml`

**#19 - Improve Memory Management**
- Auto batch size finder (binary search)
- OOM error recovery
- `--auto_batch_size` flag
- Memory profiling option
- Gradient checkpointing for large models
- `--gradient_checkpointing` flag
- Files: `src/train.py`, `src/utils.py`, `src/main.py`

**#20 - Enhance Preprocessing Pipeline**
- Custom normalization per dataset
- Grayscale and RGBA support
- Advanced augmentations:
  - Mixup
  - CutMix
  - AutoAugment
  - RandAugment
- Test-time augmentation (TTA)
- `--augmentation_policy {light,medium,heavy,custom}` flag
- `--mixup_alpha`, `--cutmix_alpha` parameters
- Files: `src/datasets.py`, `src/train.py`, `src/main.py`

**#22 - Hyperparameter Tuning Integration**
- Optuna integration
- Grid search, random search, Bayesian optimization
- `--tune` mode
- `--n_trials N` parameter
- Define search space in config or code
- Save best hyperparameters to `best_params.yaml`
- Files: NEW - `src/tune.py`, `src/main.py`

**#31 - Improve Class Imbalance Handling**
What to add:
- Automatic class weight calculation (inverse frequency)
- `--auto_weights` flag (auto-calculate from dataset)
- Focal loss implementation
- Oversampling/undersampling support
- Class-balanced sampling
- `--balance_strategy {weights,focal,oversample,undersample}` flag
- Files: `src/train.py`, `src/datasets.py`, `src/main.py`

**#34 - Add Advanced Augmentation Libraries**
What to add:
- Albumentations integration
- `--augmentation_library {torchvision,albumentations}` flag
- Pre-defined policies:
  - `light`: flip, rotation, resize
  - `medium`: + color jitter, blur, noise
  - `heavy`: + autoaugment, mixup, cutmix
  - `custom`: user-defined in config
- RandAugment implementation
- AutoAugment policies
- Files: `src/datasets.py`, `src/main.py`
- Dependencies: Add `albumentations` to requirements.txt

---

## Implementation Order (Recommended)

### Batch 1 - Foundational (Start Here)
1. **#17 - Config file support** (enables easier testing)
2. **#19 - Memory management** (prevents OOM during testing)
3. **#23 - Tests** (ensure quality as we add features)

### Batch 2 - Training Enhancements
4. **#13 - Mixed precision** (easy, big performance gain)
5. **#14 - Gradient accumulation** (easy, complements AMP)
6. **#9 - Checkpointing** (critical for long training)
7. **#21 - Multi-GPU** (performance scaling)

### Batch 3 - Evaluation & Metrics
8. **#11 - ROC/AUC/PR curves** (improves evaluation)
9. **#15 - Fix CV issues** (better validation)

### Batch 4 - Data Handling
10. **#16 - Dataset improvements** (better flexibility)
11. **#8 - Dataset download** (easier onboarding)
12. **#20 - Advanced preprocessing** (better augmentation)
13. **#34 - Augmentation libraries** (Albumentations)
14. **#31 - Class imbalance** (better performance on imbalanced data)

### Batch 5 - Optimization
15. **#22 - Hyperparameter tuning** (automatic optimization)

---

## Files to Modify

### Existing Files
- `src/main.py` - Add ~20 new CLI arguments
- `src/train.py` - Major updates (~800 lines)
- `src/datasets.py` - Add download, augmentation (~400 lines)
- `src/utils.py` - Add ROC/PR curves, memory utils (~200 lines)
- `requirements.txt` - Add: albumentations, optuna, pyyaml, pytest

### New Files to Create
- `src/config.py` - YAML/JSON config loader (~150 lines)
- `src/tune.py` - Hyperparameter tuning (~200 lines)
- `tests/test_datasets.py` (~150 lines)
- `tests/test_train.py` (~200 lines)
- `tests/test_inference.py` (~100 lines)
- `tests/test_utils.py` (~100 lines)
- `tests/conftest.py` - Pytest fixtures (~50 lines)
- `pytest.ini` - Pytest configuration
- `config_templates/basic_training.yaml`
- `config_templates/advanced_training.yaml`
- `config_templates/hyperparameter_search.yaml`

---

## Detailed Implementation Specs

### #9 - Resume/Checkpoint

**Files:** `src/train.py`, `src/main.py`

**CLI Arguments:**
```python
--resume PATH              # Path to checkpoint to resume from
--checkpoint_freq N        # Save checkpoint every N epochs (default: 10)
--checkpoint_dir PATH      # Directory to save checkpoints (default: log/exp/checkpoints/)
```

**Checkpoint Format (checkpoint_epoch_N.pth):**
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'best_loss': best_loss,
    'best_accuracy': best_accuracy,
    'history': history,
    'config': config.__dict__
}
```

**Implementation in train.py:**
```python
def save_checkpoint(self, epoch, best_loss, best_accuracy, history):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy,
        'history': history,
        'config': vars(self.config)
    }
    checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if self.scheduler and checkpoint['scheduler_state_dict']:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_loss'], checkpoint['best_accuracy'], checkpoint['history']
```

---

### #11 - ROC Curves, AUC, PR Curves

**Files:** `src/utils.py`, `src/train.py`

**New Functions in utils.py:**
```python
def plot_roc_curves(y_true, y_pred_proba, classes, save_path=None):
    """Plot ROC curves for multi-class classification"""
    # For each class: calculate FPR, TPR, AUC
    # Plot all curves on one figure
    # Save to save_path

def plot_pr_curves(y_true, y_pred_proba, classes, save_path=None):
    """Plot Precision-Recall curves for multi-class"""
    # For each class: calculate precision, recall, AP
    # Plot all curves on one figure

def calculate_auc_scores(y_true, y_pred_proba, classes):
    """Calculate AUC for each class and macro/micro average"""
    # Return dict with per-class AUC and averages
```

**Integration in train.py (evaluate method):**
```python
# After getting predictions
y_pred_proba = torch.nn.functional.softmax(all_outputs, dim=1)
y_true = all_labels

# Plot ROC curves
roc_path = os.path.join(self.config.log_path, f'roc_curve_{mode.lower()}.png')
plot_roc_curves(y_true, y_pred_proba, self.dataset.classes, roc_path)

# Plot PR curves
pr_path = os.path.join(self.config.log_path, f'pr_curve_{mode.lower()}.png')
plot_pr_curves(y_true, y_pred_proba, self.dataset.classes, pr_path)

# Calculate and save AUC scores
auc_scores = calculate_auc_scores(y_true, y_pred_proba, self.dataset.classes)
auc_path = os.path.join(self.config.log_path, f'auc_scores_{mode.lower()}.txt')
with open(auc_path, 'w') as f:
    for cls, score in auc_scores.items():
        f.write(f"{cls}: {score:.4f}\n")
```

**Dependencies:** sklearn.metrics (roc_curve, auc, precision_recall_curve, average_precision_score)

---

### #13 - Mixed Precision Training (AMP)

**Files:** `src/train.py`, `src/main.py`

**CLI Argument:**
```python
--mixed_precision    # Enable automatic mixed precision training
```

**Implementation in train.py:**
```python
# In __init__:
if self.config.mixed_precision and torch.cuda.is_available():
    self.scaler = torch.cuda.amp.GradScaler()
    self.use_amp = True
else:
    self.scaler = None
    self.use_amp = False

# In training loop:
if self.use_amp:
    with torch.cuda.amp.autocast():
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

    self.optimizer.zero_grad()
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

---

### #14 - Gradient Accumulation

**Files:** `src/train.py`, `src/main.py`

**CLI Argument:**
```python
--accumulation_steps N    # Accumulate gradients over N steps (default: 1)
```

**Implementation in train.py:**
```python
# In training loop:
for i, (inputs, labels) in enumerate(dataloader):
    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)

    # Normalize loss by accumulation steps
    loss = loss / self.config.accumulation_steps

    if self.use_amp:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only step optimizer every accumulation_steps
    if (i + 1) % self.config.accumulation_steps == 0:
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
```

---

### #15 - Fix CV Issues

**Files:** `src/train.py`

**Changes:**
1. Split data into train+val (for CV) and test (held out)
2. Perform K-fold only on train+val
3. Save per-fold history
4. Implement ensemble predictions
5. Evaluate on held-out test set at the end

**Implementation:**
```python
def run_cv(self):
    # 1. Split dataset: 80% train+val (for CV), 20% test (held out)
    # 2. For each fold in K-fold on train+val:
    #    - Train model
    #    - Save fold model and history
    #    - Collect fold predictions on validation
    # 3. Ensemble predictions across folds (voting or averaging)
    # 4. Select best fold based on validation metrics
    # 5. Load best fold model
    # 6. Evaluate on held-out test set
    # 7. Save ensemble results
```

**Output files:**
- `fold_1_history.csv`, `fold_2_history.csv`, ...
- `fold_1_best.pth`, `fold_2_best.pth`, ...
- `ensemble_predictions.csv`
- `best_fold_info.txt`
- `test_set_evaluation.txt`

---

### #21 - Multi-GPU Support

**Files:** `src/train.py`, `src/main.py`

**CLI Arguments:**
```python
--gpus IDS               # GPU IDs to use (e.g., "0,1,2,3")
--distributed            # Use DistributedDataParallel instead of DataParallel
```

**Implementation in train.py:**
```python
# In __init__:
if self.config.gpus and len(self.config.gpus.split(',')) > 1:
    gpu_ids = [int(x) for x in self.config.gpus.split(',')]
    if self.config.distributed:
        # DistributedDataParallel setup
        torch.distributed.init_process_group(backend='nccl')
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=gpu_ids)
    else:
        # DataParallel setup
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

    # Scale batch size by number of GPUs
    self.config.batch_size *= len(gpu_ids)
```

---

### #23 - Tests

**Files:** NEW - `tests/*.py`

**Directory structure:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_datasets.py         # Dataset tests
├── test_train.py            # Training tests
├── test_inference.py        # Inference tests
└── test_utils.py            # Utils tests
```

**Example tests/test_datasets.py:**
```python
import pytest
from src.datasets import ImageDataset

def test_dataset_loading():
    """Test basic dataset loading"""
    dataset = ImageDataset(...)
    assert len(dataset.classes) > 0
    assert dataset.dataloaders is not None

def test_dataset_split_ratios():
    """Test dataset split ratios"""
    dataset = ImageDataset(..., split_ratio=(0.7, 0.2, 0.1))
    # Verify splits are correct

def test_cross_validation_folds():
    """Test CV fold creation"""
    dataset = ImageDataset(..., cross_validation=True, num_splits=5)
    assert len(dataset.dataloaders) == 5
```

**pytest.ini:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**Run tests:**
```bash
pytest tests/ -v
pytest tests/test_datasets.py -v
pytest tests/ --cov=src --cov-report=html
```

---

### #8 - Dataset Download Functionality

**Files:** `src/datasets.py`, `src/main.py`

**CLI Argument:**
```python
--auto_download    # Auto-download dataset if not found
```

**Implementation in datasets.py:**
```python
DATASET_URLS = {
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
    'mnist': 'http://yann.lecun.com/exdb/mnist/',
    # Add more datasets
}

def _download(self):
    """Download dataset if URL is available"""
    if self.name.lower() not in DATASET_URLS:
        raise ValueError(f"No download URL for dataset '{self.name}'")

    url = DATASET_URLS[self.name.lower()]
    download_path = os.path.join(self.compressed_data_path, f'{self.name}.zip')

    # Download with progress bar
    from tqdm import tqdm
    import requests

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(download_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Verify hash (optional)
    # Unzip automatically
```

---

### #16 - Fix Dataset Handling Limitations

**Files:** `src/datasets.py`, `src/main.py`

**CLI Arguments:**
```python
--normalize_mean R,G,B      # Custom normalization mean (default: ImageNet)
--normalize_std R,G,B       # Custom normalization std
--support_grayscale         # Support 1-channel images
```

**Changes in datasets.py:**
1. Support absolute paths
2. Support symlinks (use `os.path.realpath()`)
3. Auto-call `_clean()` on initialization
4. Grayscale support (convert or use directly)
5. Custom normalization

```python
# In __init__:
# Support absolute paths
if os.path.isabs(self.name):
    self.images_folder_path = self.name
else:
    # Existing relative path logic

# Auto-clean corrupted images
self._clean()

# Custom normalization
if self.normalize_mean and self.normalize_std:
    mean = tuple(map(float, self.normalize_mean.split(',')))
    std = tuple(map(float, self.normalize_std.split(',')))
else:
    mean = (0.485, 0.456, 0.406)  # ImageNet defaults
    std = (0.229, 0.224, 0.225)

# Grayscale support
if self.support_grayscale:
    # Don't force RGB conversion
    # Adjust transforms for 1-channel input
```

---

### #17 - YAML/JSON Config File Support

**Files:** NEW - `src/config.py`, `config_templates/*.yaml`

**CLI Argument:**
```python
--config PATH    # Path to YAML/JSON config file
```

**src/config.py:**
```python
import yaml
import json
from typing import Dict, Any

class ConfigLoader:
    """Load and merge configuration from files and CLI"""

    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """Load config from YAML or JSON file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")

    @staticmethod
    def merge_with_cli(file_config: Dict, cli_args: Dict) -> Dict:
        """Merge file config with CLI args (CLI takes precedence)"""
        merged = file_config.copy()
        for key, value in cli_args.items():
            if value is not None:  # CLI arg was specified
                merged[key] = value
        return merged

    @staticmethod
    def save_config(config: Dict, save_path: str):
        """Save final config to file"""
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
```

**config_templates/basic_training.yaml:**
```yaml
# Basic Training Configuration
dataset_name: my_dataset
model_name: resnet50
epochs: 100
batch_size: 32
learning_rate: 0.001
image_size: 256

# Training settings
train: true
eval: true
save: true

# Augmentation
aug: true

# Splits
split_ratio: "0.8,0.1,0.1"
```

**config_templates/advanced_training.yaml:**
```yaml
# Advanced Training Configuration
dataset_name: my_dataset
model_name: efficientnet_b0
attention_name: se_layer
attention_index: 4

# Training
epochs: 100
batch_size: 16
learning_rate: 0.001
image_size: 256

# Optimization
lr_scheduler: plateau
early_stopping: 15
mixed_precision: true
accumulation_steps: 4

# Multi-GPU
gpus: "0,1,2,3"

# Checkpointing
checkpoint_freq: 10

# Augmentation
aug: true
augmentation_policy: heavy
mixup_alpha: 0.2
cutmix_alpha: 1.0

# Class imbalance
auto_weights: true
```

**Integration in main.py:**
```python
if args.config:
    file_config = ConfigLoader.load_from_file(args.config)
    final_config = ConfigLoader.merge_with_cli(file_config, vars(args))
else:
    final_config = vars(args)

# Save config to log directory
config_save_path = os.path.join(log_path, 'config_used.yaml')
ConfigLoader.save_config(final_config, config_save_path)
```

---

### #19 - Improve Memory Management

**Files:** `src/train.py`, `src/utils.py`, `src/main.py`

**CLI Arguments:**
```python
--auto_batch_size         # Automatically find optimal batch size
--gradient_checkpointing  # Enable gradient checkpointing for large models
--memory_profile          # Enable memory profiling
```

**New function in utils.py:**
```python
def find_optimal_batch_size(model, device, image_size, initial_batch_size=64):
    """Find largest batch size that fits in memory using binary search"""
    min_batch = 1
    max_batch = initial_batch_size
    optimal_batch = 1

    while min_batch <= max_batch:
        batch_size = (min_batch + max_batch) // 2
        try:
            # Test forward + backward pass
            dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()

            # Success - try larger batch
            optimal_batch = batch_size
            min_batch = batch_size + 1

            # Clean up
            del dummy_input, output, loss
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # OOM - try smaller batch
                max_batch = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e

    return optimal_batch
```

**Implementation in train.py:**
```python
# In __init__:
if self.config.auto_batch_size and torch.cuda.is_available():
    optimal_bs = find_optimal_batch_size(self.model, self.device, self.config.image_size)
    self.log(f"[INFO]  Auto-detected optimal batch size: {optimal_bs}")
    self.config.batch_size = optimal_bs

if self.config.gradient_checkpointing:
    # Enable gradient checkpointing for supported models
    if hasattr(self.model, 'gradient_checkpointing_enable'):
        self.model.gradient_checkpointing_enable()
```

**OOM Recovery:**
```python
# In training loop:
try:
    # Normal training
    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)
    loss.backward()
except RuntimeError as e:
    if 'out of memory' in str(e):
        self.log("[WARNING] OOM detected! Reducing batch size...")
        torch.cuda.empty_cache()
        self.config.batch_size = max(1, self.config.batch_size // 2)
        # Recreate dataloaders with new batch size
        # Continue training
    else:
        raise e
```

---

### #20 - Enhance Preprocessing Pipeline

**Files:** `src/datasets.py`, `src/train.py`, `src/main.py`

**CLI Arguments:**
```python
--augmentation_policy {light,medium,heavy,custom}
--mixup_alpha FLOAT           # Mixup alpha parameter (default: 0)
--cutmix_alpha FLOAT          # CutMix alpha parameter (default: 0)
--test_time_augmentation      # Enable TTA during evaluation
```

**Implementation of Mixup:**
```python
def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**Implementation of CutMix:**
```python
def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

**Augmentation Policies:**
```python
AUGMENTATION_POLICIES = {
    'light': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'medium': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'heavy': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}
```

**Test-Time Augmentation:**
```python
def predict_with_tta(model, image, num_augmentations=5):
    """Test-time augmentation for inference"""
    predictions = []

    for _ in range(num_augmentations):
        # Apply random augmentation
        augmented = apply_random_augmentation(image)
        pred = model(augmented)
        predictions.append(pred)

    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

---

### #22 - Hyperparameter Tuning Integration

**Files:** NEW - `src/tune.py`, `src/main.py`

**CLI Arguments:**
```python
--tune                    # Enable hyperparameter tuning mode
--n_trials N              # Number of tuning trials (default: 50)
--tuning_method {optuna,grid,random}
```

**src/tune.py:**
```python
import optuna
from src.train import Config, Trainer

def objective(trial):
    """Optuna objective function"""
    # Define search space
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])

    # Create config with suggested hyperparameters
    config_dict = {
        'dataset_name': base_config.dataset_name,
        'model_name': base_config.model_name,
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'epochs': 20,  # Shorter for tuning
        # ... other params
    }

    config = Config(config_dict)
    trainer = Trainer(config)

    # Train and return validation metric
    history = trainer.run()
    val_accuracy = max(history['val_accuracy'])

    return val_accuracy

def run_tuning(base_config, n_trials=50):
    """Run hyperparameter tuning"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"Best trial: {best_value}")
    print(f"Best params: {best_params}")

    # Save to file
    with open('best_params.yaml', 'w') as f:
        yaml.dump(best_params, f)

    return best_params
```

**Integration in main.py:**
```python
if args.tune:
    from src.tune import run_tuning
    best_params = run_tuning(config, n_trials=args.n_trials)
    print(f"Tuning complete! Best parameters saved to best_params.yaml")
else:
    # Normal training
    trainer = Trainer(config)
    trainer.run()
```

---

### #31 - Improve Class Imbalance Handling

**What to add:**
- Automatic class weight calculation
- Focal loss
- Oversampling/undersampling
- Class-balanced sampling

**Files:** `src/train.py`, `src/datasets.py`, `src/main.py`

**CLI Arguments:**
```python
--auto_weights                                      # Auto-calculate class weights
--balance_strategy {weights,focal,oversample,undersample}
--focal_loss_gamma FLOAT                            # Focal loss gamma (default: 2.0)
```

**Automatic Weight Calculation:**
```python
def calculate_class_weights(dataset):
    """Calculate inverse frequency weights"""
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    total = sum(class_counts.values())
    weights = {cls: total / (len(class_counts) * count)
               for cls, count in class_counts.items()}

    # Convert to tensor
    weight_tensor = torch.tensor([weights[i] for i in range(len(weights))])
    return weight_tensor
```

**Focal Loss Implementation:**
```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

**Oversampling with WeightedRandomSampler:**
```python
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    """Create sampler for oversampling minority classes"""
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    weights = [1.0 / class_counts[label] for _, label in dataset]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler
```

**Integration in train.py:**
```python
# In __init__:
if self.config.auto_weights:
    class_weights = calculate_class_weights(self.dataset.dataloaders['train'].dataset)
    self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
elif self.config.balance_strategy == 'focal':
    self.criterion = FocalLoss(gamma=self.config.focal_loss_gamma)
elif self.config.balance_strategy == 'oversample':
    # Create balanced sampler
    sampler = create_balanced_sampler(self.dataset.dataloaders['train'].dataset)
    # Recreate dataloader with sampler
```

---

### #34 - Add Advanced Augmentation Libraries

**What to add:**
- Albumentations integration
- Pre-defined policies (light/medium/heavy/custom)
- RandAugment
- AutoAugment

**Files:** `src/datasets.py`, `src/main.py`

**CLI Arguments:**
```python
--augmentation_library {torchvision,albumentations}
--augmentation_policy {light,medium,heavy,custom}
```

**Dependencies:**
```bash
pip install albumentations
```

**Albumentations Integration:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

ALBUMENTATIONS_POLICIES = {
    'light': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]),
    'medium': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.2),
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]),
    'heavy': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(p=1),
        ], p=0.3),
        A.OneOf([
            A.GridDistortion(p=1),
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
}
```

**RandAugment Implementation:**
```python
from torchvision.transforms import RandAugment

def get_randaugment_transform(image_size, mean, std, n=2, m=9):
    """RandAugment with n operations and magnitude m"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandAugment(num_ops=n, magnitude=m),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
```

**Dataset Class Update:**
```python
# In ImageDataset.__init__:
if self.augmentation_library == 'albumentations':
    if self.augmentation_policy in ALBUMENTATIONS_POLICIES:
        train_transform = ALBUMENTATIONS_POLICIES[self.augmentation_policy]
    else:
        raise ValueError(f"Unknown policy: {self.augmentation_policy}")
else:  # torchvision
    if self.augmentation_policy in AUGMENTATION_POLICIES:
        train_transform = AUGMENTATION_POLICIES[self.augmentation_policy]
    else:
        raise ValueError(f"Unknown policy: {self.augmentation_policy}")
```

---

## Dependencies to Add

**requirements.txt additions:**
```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Configuration
pyyaml>=6.0

# Augmentation
albumentations>=1.3.0

# Hyperparameter tuning
optuna>=3.0.0

# Visualization (for ROC/PR curves)
scikit-learn>=1.0.0

# Progress bars
tqdm>=4.65.0

# HTTP requests (for dataset download)
requests>=2.28.0
```

---

## Testing Strategy

After implementing each feature:

1. **Unit tests** - Test individual functions
2. **Integration tests** - Test full workflows
3. **Manual testing** - Run on real datasets
4. **Performance testing** - Measure speed/memory

**Test checklist:**
- [ ] Checkpointing: Save and resume works
- [ ] ROC/AUC: Curves generated correctly
- [ ] AMP: Training works with mixed precision
- [ ] Gradient accumulation: Effective batch size correct
- [ ] CV: Test set held out, ensemble works
- [ ] Multi-GPU: Model trains on multiple GPUs
- [ ] Config files: YAML/JSON loading works
- [ ] Auto batch size: Finds optimal size
- [ ] Mixup/CutMix: Augmentations applied correctly
- [ ] Hyperparameter tuning: Optuna integration works
- [ ] Class weights: Auto-calculation correct
- [ ] Albumentations: Library integration works

---

## Documentation Updates Needed

After implementation:

1. Update `README.md` with all new features
2. Update `NEW_FEATURES_COMPLETE.md` (or create PHASE_2_FEATURES.md)
3. Create `TESTING_GUIDE.md`
4. Create example config files in `config_templates/`
5. Update CLI help text
6. Add docstrings to all new functions
7. Create Jupyter notebook examples for new features

---

## Estimated Implementation Time

**Per feature (rough estimates):**
- #9 Checkpointing: 2-3 hours
- #11 ROC/AUC/PR: 2-3 hours
- #13 AMP: 1 hour
- #14 Gradient accumulation: 1 hour
- #15 Fix CV: 3-4 hours
- #21 Multi-GPU: 2-3 hours
- #23 Tests: 4-5 hours
- #8 Download: 2 hours
- #16 Dataset improvements: 3 hours
- #17 Config files: 2-3 hours
- #19 Memory management: 3-4 hours
- #20 Preprocessing: 4-5 hours
- #22 Hyperparameter tuning: 3-4 hours
- #31 Class imbalance: 2-3 hours
- #34 Albumentations: 2-3 hours

**Total: ~40-50 hours of development**

---

## Quick Start Guide for Next Session

**To continue implementation:**

1. Read this file: `IMPLEMENTATION_PLAN.md`
2. Start with Batch 1 (Config files, Memory, Tests)
3. Follow implementation specs above
4. Test each feature before moving to next
5. Update documentation as you go
6. Mark completed features in this file

**Command to start:**
```bash
# Read the plan
cat IMPLEMENTATION_PLAN.md

# Start with #17 - Config file support
# Create src/config.py first
```

---

**Status:** Ready to implement Phase 2 features
**Next Step:** Start with #17 (Config file support) or #23 (Tests) or #13 (AMP - easiest)
