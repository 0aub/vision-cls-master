import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from datetime import datetime
import numpy as np
import pandas as pd
import copy
import time
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from joblib import dump, load

from datasets import ImageDataset
from modules import pretrained_network, get_ml_model
from utils import cm_plot, cm_to_dict, performance_report

class Config(dict):
    def __init__(self, config):
        # Set default values
        defaults = {
            'exp': 'default',
            'root': '',
            'split_ratio': (0.8, 0.2),
            'cross_validation': False,
            'num_splits': 5,
            'model_name': 'mobilenet_v2',
            'model_type': 'dl',
            'attention_name': None,
            'attention_index': 4,
            'train': True,
            'eval': True,
            'save': True,
            'overwrite': True,
            'batch_size': 16,
            'aug': True,
            'image_size': 256,
            'learning_rate': 0.0001,
            'epochs': 100,
            'printing': True,
            'loss_weights': None
        }

        # Update defaults with provided config
        defaults.update(config)
        self.update(defaults)
        for key, value in self.items():
            setattr(self, key, value)

        if self.model_name in ['logistic_regression', 'decision_tree', 
                               'random_forest', 'svm', 'knn', 'naive_bayes', 
                               'gbm', 'adaboost', 'lda', 'qda', 'mlp']:
            self.model_type = 'ml'

        if 'resume_log_path' in config and os.path.exists(config['resume_log_path']):
            self.log_path = config['resume_log_path']
        else:
            current_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            self.log_path = os.path.join('log', f'{self.exp} {current_datetime}')

        self.history_save_path = os.path.join(self.log_path, f'history.csv')
        self.log_save_path = os.path.join(self.log_path, f'log.txt')
        self.args_save_path = os.path.join(self.log_path, f'args.txt')
        self.eval_save_path = os.path.join(self.log_path, f'eval.txt')
        self.cm_save_path = os.path.join(self.log_path, f'cm.png')
        if self.model_type == 'dl':
            self.model_save_path = os.path.join(self.log_path, 'best.pth')
        else:
            self.model_save_path = os.path.join(self.log_path, 'best.joblib')

        # Create folders if they don't exist
        os.makedirs(self.log_path, exist_ok=True)

        self.save_arguments()

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)
        self[name] = value

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def save_arguments(self):
        with open(self.args_save_path, 'w') as f:
            for key, value in self.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\n")


class Trainer:
    def __init__(self, config):
        # params
        self.config = Config(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get train and val dataloaders
        self.dataset = ImageDataset(self.config.root, 
                                    self.config.dataset_name, 
                                    self.config.batch_size, 
                                    self.config.image_size, 
                                    self.config.printing, 
                                    self.config.aug, 
                                    self.config.split_ratio, 
                                    self.config.cross_validation, 
                                    self.config.num_splits)

        # initialize the model
        if self.config.model_type == 'dl':
            self.model = pretrained_network(self.config.model_name, 
                                            self.config.attention_name, 
                                            self.config.attention_index, 
                                            len(self.dataset.classes))
            self.model.to(self.device)

            # loss function and optimizer
            if self.config.loss_weights == None:
                self.criterion = CrossEntropyLoss()
            else:
                if torch.cuda.is_available():
                    self.config.loss_weights = self.config.loss_weights.cuda()
                self.criterion = CrossEntropyLoss(weight=self.config.loss_weights)
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.model = get_ml_model(self.config.model_name)

    def run(self):
        if self.config.printing:
            self.log('*'*50)
            self.log(' '*50 + 'model name:    ' + self.config.model_name.upper())
            self.log(' '*50 + 'dataset name:  ' + self.config.dataset_name.upper())
            self.log(' '*50 + 'batch size:    ' + str(self.config.batch_size))
            self.log(' '*50 + 'learning rate: ' + str(self.config.learning_rate))
            self.log(' '*50 + 'epochs:        ' + str(self.config.epochs))
            self.log(' '*50 + 'device:        ' + str(self.device))
            if self.config.loss_weights != None:
                self.log(' '*50 + 'weighted loss: ' + 'True')
            if self.config.attention_name != None:
                self.log(' '*50 + 'attention:     ' + self.config.attention_name.upper())
                self.log(' '*50 + 'attention idx: ' + str(self.config.attention_index))
            if self.config.cross_validation:
                self.log(' '*50 + 'cv:            ' + 'True')
                self.log(' '*50 + 'cv Splits:     ' + str(self.config.num_splits))
            self.log('*'*50)

        if self.config.cross_validation:
            self.log('\n[INFO]  Running cross-validation with {self.config.num_splits} folds.')
            for fold in range(1, self.config.num_splits + 1):
                print(f"\n[INFO]  Training fold {fold}/{self.config.num_splits}")
                train_loader = self.dataset.dataloaders[fold]['train']
                val_loader = self.dataset.dataloaders[fold]['val']
                
                # Reset model to initial state if necessary
                self.model = pretrained_network(self.config.model_name, self.config.attention_name, self.config.attention_index, len(self.dataset.classes))
                self.model.to(self.device)

                # Train and evaluate the model for the current fold
                best_model_wts, history = self.train(train_loader, val_loader)
                self.model.load_state_dict(best_model_wts)
                
                # Log results, save models, or analyze per-fold performance here
                if self.config.save:
                    fold_model_path = os.path.join(self.config.log_path, f'fold_{fold}.pth')
                    torch.save(self.model, fold_model_path)
                    self.log(f'[INFO]  Model for fold {fold} saved to {fold_model_path}')
        else:
            if self.config.train:
                self.log('\n\n[INFO]  Model Training...')
                train_loader = self.dataset.dataloaders['train']
                val_loader = self.dataset.dataloaders['val']
                best_model_wts, history = self.train(train_loader, val_loader)
                self.model.load_state_dict(best_model_wts)
                if self.config.save:
                    # save model and training history
                    torch.save(self.model, self.config.model_save_path)
                    pd.DataFrame.from_dict(history).to_csv(self.config.history_save_path, index=False)
                    self.log('\n[INFO]  Saved:  ', self.config.model_save_path)
                    self.log('\n[INFO]  Saved:  ', self.config.history_save_path)

            if self.config.eval:
                print(f"[INFO]  Loading best model from {self.config.model_save_path}")
                self.model = torch.load(self.config.model_save_path, map_location=self.device)
                if len(self.config.split_ratio) == 3:
                    loader = self.dataset.dataloaders['test']
                    self.log('\n\n[INFO]  Model Evaluation using test loader...')
                else:
                    loader = self.dataset.dataloaders['val']
                    self.log('\n\n[INFO]  Model Evaluation using val loader...')
                accuracy, precision, recall, f1, loss = self.evaluate(loader)
                txt = "\n\n[INFO]  Accuracy:  {:.4f}\n[INFO]  Precision: {:.4f}\n[INFO]  Recall:    {:.4f}\n[INFO]  F1 Score:  {:.4f}\n[INFO]  Loss:      {:.4f}\n".format(accuracy, precision, recall, f1, loss)
                self.log(txt)
                with open(self.config.eval_save_path, "+w") as f:
                    f.write(txt)

                cm_plot(self.model, self.dataset, self.config.cm_save_path)


    def run_ml(self):
        self.log('*' * 50)
        self.log(f"{'':50} model name:    {self.config.model_name.upper()}")
        self.log(f"{'':50} dataset name:  {self.config.dataset_name.upper()}")
        if self.config.cross_validation:
            self.log(' '*50 + 'cv:            ' + 'True')
            self.log(' '*50 + 'cv Splits:     ' + str(self.config.num_splits))
        self.log('*' * 50)
        
        self.eval_txt = "\n[INFO]  {} Results:\n       - Accuracy: {:.4f}\n       - Precision: {:.4f}\n       - Recall: {:.4f}\n       - F1 Score: {:.4f}\n"

        if self.config.cross_validation:
            if self.config.train:
                splits = ['train', 'val']
                x, y = [], []
                for sp in splits:
                    f, l = self.dataset.prepare_set_ml(sp)
                    x.append(f)
                    y.append(l)
                x = np.vstack(x)
                y = np.concatenate(y)
                skf = StratifiedKFold(n_splits=self.config.num_splits)
                fold_results = []
                
                for fold, (train_index, test_index) in enumerate(skf.split(x, y), start=1):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    self.model.fit(x_train, y_train)
                    accuracy, precision, recall, f1 = self.evaluate_ml(f"Fold {fold} val", x_test, y_test)
                    
                    fold_results.append((accuracy, precision, recall, f1))
                    if self.config.save:
                        fold_model_path = os.path.join(self.log_path, f'fold_{fold}.joblib')
                        dump(self.model, fold_model_path)
                        self.log(f"\n[INFO]  Fold {fold} saved: {fold_model_path}")

                # Optional: Calculate and print average metrics across all folds
                avg_accuracy = np.mean([f[0] for f in fold_results])
                avg_precision = np.mean([f[1] for f in fold_results])
                avg_recall = np.mean([f[2] for f in fold_results])
                avg_f1 = np.mean([f[3] for f in fold_results])
                self.log(self.eval_txt.format("Cross-Validation Summary - Avg Metrics", avg_accuracy, avg_precision, avg_recall, avg_f1))
            
            if self.config.eval:
                eval_results = []
                if len(self.config.split_ratio) == 3:
                    x_test, y_test = self.prepare_set_ml('test')
                    self.log('\n\n[INFO]  Model Evaluation using test set...')
                else:
                    x_test, y_test = self.prepare_set_ml('val')
                    self.log('\n\n[INFO]  Model Evaluation using val set...')
                
                # Load and evaluate each fold's model
                for fold in range(1, self.config.num_splits + 1):
                    fold_model_path = os.path.join(self.log_path, f'fold_{fold}.joblib')
                    self.model = load(fold_model_path)
                    accuracy, precision, recall, f1 = self.evaluate_ml(f"Fold {fold} Evaluation", x_test, y_test)
                    eval_results.append((accuracy, precision, recall, f1))

                # Optional: Calculate and print average metrics across all evaluations
                avg_accuracy = np.mean([result[0] for result in eval_results])
                avg_precision = np.mean([result[1] for result in eval_results])
                avg_recall = np.mean([result[2] for result in eval_results])
                avg_f1 = np.mean([result[3] for result in eval_results])
                summary_text = self.eval_txt.format("Cross-Validation Evaluation Summary - Avg Metrics", avg_accuracy, avg_precision, avg_recall, avg_f1)
                self.log(summary_text)
                with open(self.config.eval_save_path, "w") as f:
                    f.write(summary_text)
        else:
            if self.config.train:
                X_train, y_train = self.dataset.prepare_set_ml('train')
                X_val, y_val = self.dataset.prepare_set_ml('val')

                # Training
                self.log('\n[INFO]  Training started...')
                start_time = time.time()
                self.model.fit(X_train, y_train)
                self.log(f'\n[INFO]  Training ended in {str(int(time.time() - start_time))} seconds.')

                if self.config.save:
                    dump(self.model, self.config.model_save_path)
                    self.log('\n[INFO]  Saved:  ', self.config.model_save_path)
        
                # Evaluate on training set
                self.evaluate_ml("Training", X_train, y_train)
            
            if self.config.eval:
                # Evaluate on validation set
                self.evaluate_ml("Validation", X_val, y_val)

                # Evaluate on testing set
                if 'test' in self.dataset.splits:
                    X_test, y_test = self.prepare_ml_data('test')
                    self.evaluate_ml("Training", X_test, y_test)
                    


    def train(self, train_loader, val_loader):
        # history dict
        history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }
        # initial variables
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_precision = 0.0
        best_epoch = 0
        # train/val variables
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        # main training/validation loop
        for epoch in range(self.config.epochs):
            self.log("Epoch {}/{}".format(epoch+1, self.config.epochs))
            # reset epoch accuracy and loss
            epoch_timer = time.time()
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            # set model training status for the training
            self.model.train(True)
            # training iterations
            for i, data in enumerate(train_loader):
                self.log('\r\t{}/{}  time: {}s  '.format(i+1 , len(train_loader), int(time.time() - epoch_timer)), end='')
                # extract images and labels
                inputs, labels = data
                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                # Clear gradients
                self.optimizer.zero_grad()
                # predictions
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # compute loss and back propagation
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # calculate training loss and accuracy
                loss_train += loss.item()
                acc_train += torch.sum(preds == labels.data)
                # free some memory
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            # average training loss and accuracy
            # * 2 as we only used half of the dataset
            avg_loss = loss_train / self.dataset.dataset_sizes['train']
            avg_acc = acc_train / self.dataset.dataset_sizes['train']
            # change model training status for the evaluation
            self.model.train(False)
            self.model.eval()
            cm = torch.zeros(len(self.dataset.classes), len(self.dataset.classes))
            # validation iterations
            for i, data in enumerate(val_loader):
                # extract images and labels
                inputs, labels = data
                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                # Clear gradients
                self.optimizer.zero_grad()
                # predictions
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # compute loss
                loss = self.criterion(outputs, labels)
                # calculate training loss and accuracy
                loss_val += loss.item()
                acc_val += torch.sum(preds == labels.data)
                # calculate confusion matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    cm[t.long(), p.long()] += 1
                # free some memory
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            # average validation loss and accuracy
            avg_loss_val = loss_val / self.dataset.dataset_sizes['val']
            avg_acc_val = acc_val / self.dataset.dataset_sizes['val']
            # calculate precision, recall, and f1
            cm_dict = cm_to_dict(cm, self.dataset.classes)
            precision, recall, f1, _ = performance_report(cm_dict, mode = 'Macro')
            epoch_time = int(time.time() - epoch_timer)
            # logging
            self.log("loss: {:.4f}  acc: {:.4f}  val_loss: {:.4f} val_acc: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f}".format(avg_loss, avg_acc, avg_loss_val, avg_acc_val, precision, recall, f1))
            # update best accuracy
            best_acc, best_precision, best_model_wts, best_epoch = self.update_best_wts(avg_acc_val, best_acc, precision, best_precision, best_model_wts, best_epoch, epoch)

            # save progress in history
            history['epoch'].append(epoch+1)
            history['loss'].append(round(np.float64(avg_loss).item(), 4))
            history['accuracy'].append(round(np.float64(avg_acc).item(), 4))
            history['val_loss'].append(round(np.float64(avg_loss_val).item(), 4))
            history['val_accuracy'].append(round(np.float64(avg_acc_val).item(), 4))
            history['precision'].append(round(np.float64(precision).item(), 4))
            history['recall'].append(round(np.float64(recall).item(), 4))
            history['f1'].append(round(np.float64(f1).item(), 4))
            history['time'].append(epoch_time)

        # calculate training time
        self.log("\n[INFO]  Training completed in {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60))
        self.log("[INFO]  Best accuracy: {:.4f} at epoch {}".format(best_acc, best_epoch))
        
        return best_model_wts, history

    def evaluate(self, test_loader):
        # initial variables
        since = time.time()
        loss_test = 0
        acc_test = 0
        cm = torch.zeros(len(self.dataset.classes), len(self.dataset.classes))
        # testing iterations
        for i, data in enumerate(test_loader):
            # set model training status to False for the evaluation
            self.model.train(False)
            self.model.eval()
            # extract images and labels
            inputs, labels = data
            inputs = inputs.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.long)
            # predictions
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # calculate loss
            loss = self.criterion(outputs, labels)
            # calculate testing loss and accuracy
            loss_test += loss.item()
            acc_test += torch.sum(preds == labels.data)
            # calculate confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
            # free some memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # average testing loss and accuracy
        loss_test = loss_test / self.dataset.dataset_sizes['val']
        acc_test = acc_test / self.dataset.dataset_sizes['val']

        # calculate precision, recall, and f1
        cm_dict = cm_to_dict(cm, self.dataset.classes)
        precision, recall, f1, _ = performance_report(cm_dict, mode = 'Macro')

        # calculate training time
        elapsed_time = time.time() - since
        self.log("\n[INFO]  Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        return round(acc_test.item(), 4), round(precision.item(), 4), round(recall.item(), 4), round(f1.item(), 4), round(loss_test, 4)
    
    def evaluate_ml(self, set_name, X, y):
        preds = self.model.predict(X)
        accuracy = self.model.score(X, y)
        cm = confusion_matrix(y, preds)
        cm_dict = cm_to_dict(cm, self.dataset.classes)
        precision, recall, f1, _ = performance_report(cm_dict, mode='macro', printing=False)
        result_text = self.eval_txt.format(set_name, accuracy, precision, recall, f1)
        self.log(f'\n[INFO]  Evaluate on {set_name} set...')
        self.log(result_text)
        with open(self.config.eval_save_path, "w") as f:
            f.write(result_text)
        return (accuracy, precision, recall, f1)

    def log(self, text='', end='\n'):
        # Check if text starts with '\r'
        if text.startswith('\r'):
            # Move cursor to the beginning of the line
            print('\r', end='', flush=True)
        # Print text to console
        if self.config.printing:
            print(text, end=end, flush=True)
        # Save text to file
        with open(self.config.log_save_path, 'a') as f:
            f.write(text + end)

    def update_best_wts(self, avg_acc_val, best_acc, precision, best_precision, best_model_wts, best_epoch,  epoch):
        if avg_acc_val > best_acc:
            self.log("\tval_acc improved from {:.4f} to {:.4f}".format(best_acc, avg_acc_val))
            best_acc = avg_acc_val
            best_precision = precision
            best_epoch = epoch
            best_model_wts = copy.deepcopy(self.model.state_dict())
        elif avg_acc_val == best_acc and precision > best_precision:
            self.log("\tval_precision improved from {:.4f} to {:.4f}".format(best_precision, precision))
            best_precision = precision
            best_epoch = epoch
            best_model_wts = copy.deepcopy(self.model.state_dict())
        else:
            self.log("\tval_acc did not improve from {:.4f} (epoch: {})".format(best_acc, best_epoch))

        return best_acc, best_precision, best_model_wts, best_epoch