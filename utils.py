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