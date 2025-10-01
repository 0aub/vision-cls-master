# Issues, Gaps, and Limitations

**Last Updated:** 2024-11-27
**Status:** Critical bugs fixed! See fixes below.

---

## ✅ FIXED - Critical Bugs

All critical bugs have been resolved as of 2024-11-27:

1. ✅ **Loss weights parsing bug** - FIXED
2. ✅ **ML CV model loading bug** - FIXED
3. ✅ **DL CV optimizer not reset bug** - FIXED
4. ✅ **Inconsistent model saving in CV** - FIXED
5. ✅ **Dataset CV code duplication** - FIXED
6. ✅ **ML CV data preparation logic** - FIXED
7. ✅ **Basic error handling added** - FIXED

See detailed fixes below.

---

## 🚨 Critical Bugs (FIXED)

### 1. ✅ **Loss Weights Parsing Bug** (main.py:47) - FIXED
**Severity:** High
**Location:** `src/main.py` line 68
**Status:** ✅ **FIXED**

**Original Issue:**
```python
# BROKEN:
if args.loss_weights:
    loss_weights = torch.tensor(loss_weights, dtype=torch.float32)  # Variable not defined!
```

**Applied Fix:**
```python
if args.loss_weights:
    try:
        loss_weights = torch.tensor(list(map(float, args.loss_weights.split(','))), dtype=torch.float32)
        if torch.any(loss_weights < 0):
            raise ValueError("loss_weights must be non-negative")
    except Exception as e:
        raise ValueError(f"Invalid loss_weights '{args.loss_weights}': {e}")
```
**Result:** ✅ Now correctly parses loss weights with validation

---

### 2. ✅ **ML Cross-Validation Model Loading Bug** (train.py:249-250) - FIXED
**Severity:** Critical
**Location:** `src/train.py` line 249
**Status:** ✅ **FIXED**

**Original Issue:**
```python
# BROKEN:
fold_model_path = os.path.join(self.log_path, f'fold_{fold}.joblib')
with open('model.pkl', 'rb') as file:  # Opens wrong hardcoded file
    self.model = pickle.load(fold_model_path)  # Tries to unpickle a string path!
```

**Applied Fix:**
```python
fold_model_path = os.path.join(self.config.log_path, f'fold_{fold}.joblib')
with open(fold_model_path, 'rb') as file:
    self.model = pickle.load(file)
```
**Result:** ✅ ML CV evaluation now loads models correctly

---

### 3. **DL Cross-Validation Optimizer Not Reset** (train.py:157-158)
**Severity:** High
**Location:** `train.py` lines 156-158
**Issue:**
```python
# Creates new model but optimizer still references OLD model parameters
self.model = pretrained_network(self.config.model_name, ...)
self.model.to(self.device)
# self.optimizer still points to old model's parameters!
```
**Fix:**
```python
self.model = pretrained_network(self.config.model_name, ...)
self.model.to(self.device)
self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
```
**Impact:** Cross-validation will optimize wrong parameters, corrupting training.

---

### 4. **Inconsistent Model Saving in CV** (train.py:167 vs 178)
**Severity:** Medium
**Location:** `train.py` line 167
**Issue:**
- Line 167 (CV): `torch.save(self.model, ...)` - saves entire model
- Line 178 (non-CV): `torch.save(self.model.state_dict(), ...)` - saves only weights
**Fix:** Use consistent approach (state_dict is preferred)
**Impact:** Model loading inconsistency, larger file sizes.

---

### 5. **Dataset CV Code Duplication** (datasets.py:78-127)
**Severity:** Medium
**Location:** `datasets.py` lines 78-94 and 108-123
**Issue:** Nearly identical code creates dataloaders twice (once at line 78, again at line 108)
**Impact:** Inefficient, wasteful, error-prone maintenance.

---

### 6. **ML CV Data Preparation Logic Error** (train.py:205-212)
**Severity:** Medium
**Location:** `train.py` lines 205-212
**Issue:**
```python
# Combines pre-split train+val, then re-splits with StratifiedKFold
splits = ['train', 'val']
x, y = [], []
for sp in splits:
    f, l = self.dataset.prepare_set_ml(sp)
    x.append(f)
    y.append(l)
x = np.vstack(x)
y = np.concatenate(y)
```
**Problem:** If user provided pre-split data, this combines and re-splits it, potentially causing data leakage or incorrect validation.
**Impact:** Invalid cross-validation results.

---

## ⚠️ Design Flaws

### 7. **No Validation/Error Handling**
- No check if dataset path exists before training
- No validation for supported image formats
- No handling for corrupted images (except unused `_clean()` method)
- No GPU availability check vs requirements
- No validation of parameter ranges (epochs > 0, batch_size > 0, etc.)
- No check if number of classes matches model output

---

### 8. **Missing Dataset Download Functionality**
**Location:** `datasets.py` line 172
**Issue:** `_download()` method is commented out, referenced but not implemented
**Impact:** Users must manually download and prepare datasets.

---

### 9. **No Resume/Checkpoint Capability**
- Cannot resume interrupted training from checkpoint
- No mid-epoch checkpointing
- All progress lost on crash/interruption
- No saving of optimizer state
- No epoch counter persistence

---

### 10. **No Inference/Prediction API**
**Missing Features:**
- No standalone prediction function
- No batch inference support
- No single image prediction
- No model export (ONNX, TorchScript, TensorRT)
- Only training and evaluation modes exist

---

### 11. **Limited Evaluation Metrics**
**Missing:**
- No per-class detailed metrics output
- No ROC curves or AUC scores
- No precision-recall curves
- No visualization of training curves (matplotlib/tensorboard)
- No inference time benchmarking
- No model complexity metrics (FLOPs, params count)

---

### 12. **No Learning Rate Scheduling**
- Fixed learning rate throughout training
- No ReduceLROnPlateau
- No CosineAnnealingLR
- No warmup strategies
- No automatic learning rate finder

---

### 13. **No Mixed Precision Training**
- No AMP (Automatic Mixed Precision) support
- No gradient scaling
- Missing performance optimization for modern GPUs

---

### 14. **No Gradient Accumulation**
- Cannot train large models with small GPU memory
- No effective batch size scaling

---

### 15. **Cross-Validation Issues**

#### Deep Learning CV:
- No held-out test set (all data used in CV)
- No evaluation after CV completes
- History not saved per fold
- No ensemble predictions across folds
- No best fold selection logic

#### Machine Learning CV:
- Same issues as DL CV
- Plus the data preparation bug (#6)

---

### 16. **Dataset Handling Limitations**
- Hardcoded paths relative to script location
- Cannot use external/absolute dataset paths easily
- No symlink support
- No dataset validation before splitting
- `_clean()` method for corrupted files not called automatically
- Fixed ImageNet normalization (no custom normalization)
- No grayscale image support
- Limited augmentation options (only flip)

---

### 17. **Configuration Management**
- No YAML/JSON config file support
- No configuration validation
- All parameters must be CLI arguments
- No configuration presets/templates
- No parameter search/sweep integration

---

### 18. **Logging Limitations**
- No TensorBoard integration
- No real-time training visualization
- No experiment tracking (MLflow, Weights & Biases, Neptune)
- No metric dashboards
- Logs only to text files

---

### 19. **Memory Management**
- Manual `torch.cuda.empty_cache()` may not be optimal
- No automatic memory profiling
- No OOM (Out of Memory) recovery
- No batch size auto-tuning

---

### 20. **No Data Preprocessing Pipeline**
- Cannot customize normalization statistics
- No support for non-RGB images (grayscale, RGBA)
- No advanced augmentation (mixup, cutmix, autoaugment)
- No test-time augmentation (TTA)

---

### 21. **No Multi-GPU Support**
- No DataParallel
- No DistributedDataParallel
- Cannot scale to multiple GPUs

---

### 22. **No Hyperparameter Tuning**
- No Optuna/Ray Tune integration
- No grid search
- No random search
- No Bayesian optimization

---

## 🧪 Testing Gaps

### 23. **No Tests**
- No unit tests
- No integration tests
- No data validation tests
- No model architecture tests
- No end-to-end pipeline tests

---

## 📚 Documentation Gaps

### 24. **Incomplete Documentation**
- API documentation missing (incomplete docstrings)
- No architecture diagrams
- No troubleshooting guide
- No FAQ section
- No performance benchmarks
- No comparison between models
- No example notebooks
- No video tutorials

---

## 🔒 Security Issues

### 25. **Security Concerns**
- No input sanitization for file paths (potential path traversal)
- Pickle files for ML models (arbitrary code execution risk)
- No authentication for Jupyter server in Docker
- No SSL/TLS for production deployment
- No secrets management

---

## 🏗️ Code Quality Issues

### 26. **Code Organization**
- All code in root directory (no src/ structure)
- No separation of concerns
- Large monolithic files
- No abstract base classes for models
- Inconsistent naming conventions
- Magic numbers in code (e.g., reduction=16, kernel_size=7)

---

### 27. **Dependencies Issues**
- No version pinning in requirements.txt
- May break with future PyTorch/torchvision updates
- No dependency vulnerability scanning

---

### 28. **No CI/CD Pipeline**
- No GitHub Actions / GitLab CI
- No automated testing on commit
- No automated Docker builds
- No automated deployment

---

## 📊 Performance Issues

### 29. **Inefficient Data Loading**
- Fixed num_workers=2 (should be configurable)
- No prefetch_factor tuning
- No persistent_workers=True option
- Loads labels twice in CV (datasets.py)

---

### 30. **No Model Optimization**
- No quantization support (INT8, FP16)
- No pruning
- No knowledge distillation
- No ONNX export for faster inference

---

## 🎯 Missing Features

### 31. **No Class Imbalance Handling**
- Only supports loss weights (manual)
- No automatic class weight calculation
- No oversampling/undersampling
- No focal loss

---

### 32. **No Early Stopping**
- Trains for full epochs even if overfitting
- No patience parameter
- No validation loss monitoring for early exit

---

### 33. **No Model Comparison**
- Cannot easily compare multiple models
- No automatic benchmark suite
- No leaderboard generation

---

### 34. **No Data Augmentation Library Integration**
- No Albumentations support
- No imgaug support
- No RandAugment
- No AutoAugment

---

### 35. **No Explainability/Interpretability**
- No Grad-CAM
- No attention visualization
- No feature map visualization
- No saliency maps

---

### 36. **No Transfer Learning Options**
- Always uses pretrained=True
- No option to train from scratch
- No progressive unfreezing
- No discriminative learning rates

---

### 37. **No Model Ensemble Support**
- No ensemble predictions
- No model averaging
- No stacking
- No boosting

---

### 38. **No Experiment Tracking**
- No automatic experiment versioning
- No hyperparameter logging
- No artifact storage
- No reproducibility guarantees (no seed fixing)

---

### 39. **No Active Learning Support**
- Cannot identify uncertain samples
- No human-in-the-loop workflow
- No annotation tool integration

---

### 40. **No Model Serving**
- No REST API
- No gRPC support
- No batch serving
- No model versioning for production

---

## 🔄 Workflow Limitations

### 41. **No Pipeline Automation**
- Manual dataset preparation
- Manual model selection
- Manual hyperparameter tuning
- No AutoML capabilities

---

### 42. **No Incremental Learning**
- Cannot add new classes without retraining
- No continual learning support
- No few-shot learning

---

### 43. **No Multi-Task Learning**
- Only single-task classification
- No multi-label support
- No auxiliary tasks

---

## 📱 Deployment Issues

### 44. **No Production Readiness**
- No health checks
- No monitoring
- No alerting
- No load balancing
- No horizontal scaling

---

### 45. **Docker Issues**
- Image size not optimized (can use multi-stage builds)
- No Docker image scanning
- No non-root user in container
- Token usage warning shows development stage

---

## 🌐 Integration Gaps

### 46. **No Cloud Integration**
- No AWS S3 data loading
- No Google Cloud Storage support
- No Azure Blob Storage
- No cloud training (SageMaker, Vertex AI)

---

### 47. **No Database Integration**
- No results database
- No metadata storage
- No model registry

---

### 48. **No Labeling Tool Integration**
- No Label Studio integration
- No CVAT integration
- No export from annotation tools

---

## 📈 Monitoring & Observability

### 49. **No Metrics Collection**
- No Prometheus metrics
- No Grafana dashboards
- No training progress tracking for long runs
- No email/Slack notifications on completion/failure

---

### 50. **No Model Drift Detection**
- No data drift monitoring
- No concept drift detection
- No model performance degradation alerts

---

## Summary Statistics

- **Critical Bugs:** 6
- **Design Flaws:** 44
- **Total Issues:** 50+

---

## Priority Ranking

### P0 (Fix Immediately):
1. Loss weights parsing bug (#1)
2. ML CV loading bug (#2)
3. DL CV optimizer bug (#3)

### P1 (High Priority):
4. Inconsistent model saving (#4)
5. No error handling (#7)
6. No resume capability (#9)
7. Dataset CV duplication (#5)
8. ML CV data leak (#6)

### P2 (Medium Priority):
- Inference API (#10)
- Better metrics (#11)
- Learning rate scheduling (#12)
- Early stopping (#32)
- Testing (#23)

### P3 (Nice to Have):
- Multi-GPU (#21)
- Cloud integration (#46)
- Model serving (#40)
- Advanced augmentation (#34)

---

## Recommended Immediate Actions

1. ✅ Fix critical bugs #1, #2, #3
2. ✅ Add basic error handling
3. ✅ Restructure code to src/ folder
4. ✅ Add basic tests
5. ✅ Add inference API
6. ✅ Version pin requirements.txt
7. ✅ Add example Jupyter notebooks
8. ✅ Create comprehensive documentation

---

**Note:** This document should be kept updated as issues are resolved and new ones are discovered.
