import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def performance_report(cm, mode='macro', printing=False):
    col = len(cm)
    labels = list(cm.keys())

    arr = []
    for key, value in cm.items():
        arr.append(value)

    cr = dict()
    support_sum = 0

    macro = [0] * 3
    weighted = [0] * 3
    for i in range(col):
        vertical_sum = sum([arr[j][i] for j in range(col)])
        horizontal_sum = sum(arr[i])

        # Safely calculate precision to avoid division by zero.
        p = arr[i][i] / vertical_sum if vertical_sum != 0 else 0

        # Safely calculate recall to avoid division by zero.
        r = arr[i][i] / horizontal_sum if horizontal_sum != 0 else 0

        # Safely calculate F1-score to avoid division by zero.
        f = (2 * p * r) / (p + r) if (p + r) != 0 else 0

        s = horizontal_sum
        row = [p, r, f, s]

        support_sum += s

        for j in range(3):
            macro[j] += row[j]
            weighted[j] += row[j] * s

        cr[i] = row

    truepos = sum(arr[i][i] for i in range(col))
    total = sum(sum(arr[i]) for i in range(col))
    cr['Accuracy'] = ["", "", truepos / total, support_sum]

    macro_avg = [Sum / col for Sum in macro]
    macro_avg.append(support_sum)
    cr['Macro_avg'] = macro_avg

    weighted_avg = [Sum / support_sum for Sum in weighted] if support_sum != 0 else [0, 0, 0]
    weighted_avg.append(support_sum)
    cr['Weighted_avg'] = weighted_avg

    if printing:
        stop = 0
        max_key = max(len(str(x)) for x in list(cr.keys())) + 15
        print("Performance report of the model is :")
        print(f"%{max_key}s %9s %9s %9s %9s\n" % (" ", "Precision", "Recall", "F1-Score", "Support"))
        for i, (key, value) in enumerate(cr.items()):
            if stop < col:
                stop += 1
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            elif stop == col:
                stop += 1
                print(f"\n%{max_key}s %9s %9s %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))
            else:
                print(f"%{max_key}s %9.2f %9.2f %9.2f %9d" % (labels[key] if isinstance(key, int) else key, value[0], value[1], value[2], value[3]))

    if mode == 'macro':
        return cr['Macro_avg']
    else:
        return cr['Weighted_avg']


def cm_to_dict(cm, labels):
    cm_dict = dict()
    for i, row in enumerate(cm):
        # The index i corresponds to the ith class label
        # The row of the confusion matrix corresponding to the label is added to the dictionary
        cm_dict[labels[i]] = row
    return cm_dict

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, cm_save_path=None):
    if torch.is_tensor(cm):
        cm = cm.cpu().numpy() 
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    if cm_save_path:
        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')

    plt.close()

def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves

    Args:
        history: Dictionary with keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = history['epoch']

    # Plot Loss
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO]  Training curves saved to {save_path}")

    plt.close()

def plot_metrics_curves(history, save_path=None):
    """
    Plot precision, recall, and F1 curves

    Args:
        history: Dictionary with keys 'precision', 'recall', 'f1'
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']

    ax.plot(epochs, history['precision'], 'g-', label='Precision', linewidth=2, marker='o')
    ax.plot(epochs, history['recall'], 'b-', label='Recall', linewidth=2, marker='s')
    ax.plot(epochs, history['f1'], 'r-', label='F1 Score', linewidth=2, marker='^')

    ax.set_title('Validation Metrics Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO]  Metrics curves saved to {save_path}")

    plt.close()

def get_per_class_metrics(cm, classes):
    """
    Calculate per-class precision, recall, and F1 scores

    Args:
        cm: Confusion matrix (numpy array or torch tensor)
        classes: List of class names

    Returns:
        DataFrame with per-class metrics
    """
    if torch.is_tensor(cm):
        cm = cm.cpu().numpy()

    num_classes = len(classes)
    metrics = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()

        metrics.append({
            'Class': classes[i],
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Support': int(support)
        })

    df = pd.DataFrame(metrics)
    return df

def print_per_class_metrics(cm, classes):
    """Print per-class metrics in a formatted table"""
    df = get_per_class_metrics(cm, classes)
    print("\n" + "="*70)
    print("Per-Class Performance Metrics")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    return df

def get_model_summary(model, input_size=(3, 256, 256)):
    """
    Get model summary with parameter count and memory estimate

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)

    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Estimate model size in MB
    param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32

    # Get model name
    model_name = model.__class__.__name__

    summary = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': param_size_mb,
        'input_size': input_size
    }

    return summary

def print_model_summary(model, input_size=(3, 256, 256)):
    """
    Print formatted model summary

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    summary = get_model_summary(model, input_size)

    print("\n" + "="*70)
    print("Model Summary")
    print("="*70)
    print(f"Model Name:           {summary['model_name']}")
    print(f"Input Size:           {summary['input_size']}")
    print(f"-" * 70)
    print(f"Total Parameters:     {summary['total_params']:,}")
    print(f"Trainable Parameters: {summary['trainable_params']:,}")
    print(f"Non-trainable Params: {summary['non_trainable_params']:,}")
    print(f"-" * 70)
    print(f"Estimated Size:       {summary['model_size_mb']:.2f} MB")
    print("="*70 + "\n")

    return summary