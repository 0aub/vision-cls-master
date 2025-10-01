# Vision-CLS Notebooks

This directory contains Jupyter notebooks for exploring and using the Vision Classification framework.

## Available Notebooks

### 1. Getting Started (`01_getting_started.ipynb`)
- Basic setup and configuration
- Simple training example
- GPU availability check
- List of available models

### 2. Attention Mechanisms (`02_attention_mechanisms.ipynb`)
- Using different attention mechanisms (SE, CBAM, ECA, etc.)
- Comparing multiple attention mechanisms
- Best practices for attention integration

### 3. Cross-Validation (`03_cross_validation.ipynb`)
- K-fold cross-validation for deep learning models
- Analyzing cross-validation results
- Cross-validation with attention mechanisms

## How to Use

1. **Start the Docker container:**
   ```bash
   ./run.sh
   ```

2. **Access Jupyter Lab:**
   Open your browser to `http://localhost:8888`

3. **Navigate to the notebooks folder** and open any notebook

4. **Prepare your dataset:**
   - Place your dataset in `data/compressed/` or `data/uncompressed/`
   - Update the `dataset_name` parameter in the notebooks

5. **Run the cells** to train your models

## Creating Your Own Notebooks

Feel free to create your own notebooks in this directory. Your notebooks will be:
- Automatically mounted in the Docker container
- Accessible via Jupyter Lab
- Persistent across container restarts

## Tips

- **GPU Memory:** Monitor GPU memory usage if training large models
- **Saving Results:** All training results are saved in the `log/` directory
- **Dataset Path:** Make sure to update `dataset_name` to match your dataset
- **Experiment Names:** Use descriptive experiment names to organize results

## Troubleshooting

### Kernel Dies During Training
- Reduce batch size
- Use a smaller model
- Check GPU memory with `nvidia-smi`

### Import Errors
- Make sure `sys.path.append('/app')` is at the top of your notebook
- Restart the kernel if you modify source files

### Dataset Not Found
- Check that your dataset is in the correct location
- Verify the dataset name matches the folder name
- Ensure the dataset is properly structured (class subfolders)

## Example Workflow

```python
# 1. Import
import sys
sys.path.append('/app')
from src.train import Config, Trainer

# 2. Configure
config_dict = {
    'dataset_name': 'my_dataset',
    'model_name': 'resnet50',
    'epochs': 20
}

# 3. Train
config = Config(config_dict)
trainer = Trainer(config)
trainer.run()

# 4. Check results in log/ directory
```

Happy experimenting!
