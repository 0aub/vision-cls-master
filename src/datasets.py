import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import numpy as np
import os

class ImageDataset():
    def __init__(self, root='', dataset_name='ucmerced', batch_size=16, image_size=256, printing=False, aug=True, split_ratio=(0.8, 0.2), cross_validation=False, num_splits=5):
        # fixed paths
        base_download_path = os.path.join(root, 'data/compressed/')
        base_images_folder_path = os.path.join(root, 'data/uncompressed/')
        base_splitted_data_path = os.path.join(root, 'data/splitted/')

        # prepare the class params
        self.aug = aug
        dataset_name = dataset_name
        self.images_folder_path = os.path.join(base_images_folder_path, dataset_name)

        self.name = dataset_name
        self.zip_file = self.name + '.zip'
        self.download_path = base_download_path
        self.unzip_path = base_images_folder_path
        self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
        self.length = self._count_datafiles()

        self.split_ratio = split_ratio
        if len(self.split_ratio) == 2:
            self.splits = ['train', 'val']
        elif len(self.split_ratio) == 3:
            self.splits = ['train', 'val', 'test']
        else:
            raise ValueError(f'[ERROR]  wrong split ratio: {self.split_ratio}')
        
        self.cross_validation = cross_validation
        self.num_splits = num_splits

        self.dataloaders, self.dataset_sizes, self.classes = self._get_data(batch_size, image_size, printing)

    def _get_data(self, batch_size, image_size, printing):
        # Check for data availability
        if not os.path.exists(os.path.join(self.splitted_data_path, 'train')):
            # Check if source data exists before attempting to split
            if not os.path.exists(self.images_folder_path):
                raise FileNotFoundError(
                    f"Dataset not found at {self.images_folder_path}. "
                    f"Please place your dataset in data/compressed/{self.name}.zip "
                    f"or data/uncompressed/{self.name}/"
                )
            self._split()

        # Validate that train folder has data
        train_path = os.path.join(self.splitted_data_path, 'train')
        if not os.listdir(train_path):
            raise ValueError(f"Training data folder is empty: {train_path}")

        # self._clean()

        # transforms (data augmentation)
        imagenet_normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(image_size),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip() if self.aug else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip() if self.aug else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_normalization)
            ]),
            'val': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_normalization)
            ]),
            'test': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_normalization)
            ])
        }
        # initialize dataseta
        image_datasets = {
            x: ImageFolder(
                os.path.join(self.splitted_data_path, x),
                transform=data_transforms[x]
            ) for x in self.splits
        }

        if self.cross_validation:
            # Ensure labels are collected from the dataset for stratification
            labels = [y for _, y in DataLoader(image_datasets['train'], batch_size=1)]
            labels = np.array(labels)  # Convert list of labels to a numpy array

            # Instantiate the StratifiedKFold object
            skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=42)

            # Initialize dataloaders dictionary to hold train and validation loaders for each fold
            dataloaders = {fold: {'train': None, 'val': None} for fold in range(1, self.num_splits + 1)}

            # Populate the dataloaders dictionary with data loaders for each fold
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                dataloaders[fold]['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=2)
                dataloaders[fold]['val'] = DataLoader(image_datasets['train'], batch_size=batch_size, sampler=val_sampler, num_workers=2)

                # Print fold information if printing is enabled
                if printing:
                    print(f"[INFO]  Fold {fold}:")
                    print(f"\tLoaded {len(train_sampler.indices)} images for train")
                    print(f"\tLoaded {len(val_sampler.indices)} images for val")

                    # Further break down into classes if needed
                    print("[INFO]  Class distribution in this fold:")
                    train_class_counts = {cls: 0 for cls in image_datasets['train'].classes}
                    for _, label in DataLoader(image_datasets['train'], batch_size=1, sampler=train_sampler):
                        train_class_counts[image_datasets['train'].classes[label.item()]] += 1
                    for cls, count in train_class_counts.items():
                        print(f"\t\t{cls.ljust(20)} -> count: {count}")
        else:
            dataloaders = {
                x: DataLoader(
                    image_datasets[x], batch_size=batch_size,
                    shuffle=True, num_workers=2
                ) for x in self.splits
            }

        # printing
        dataset_sizes = {x: len(image_datasets[x]) for x in self.splits}
        classes = image_datasets['train'].classes

        if printing and not self.cross_validation:
            for x in self.splits:
                print("[INFO]  Loaded {} images under {}".format(dataset_sizes[x], x))
            print("[INFO]  Classes: ")
            for cls in classes:
                train_count = len(os.listdir(os.path.join(self.splitted_data_path, "train", cls)))
                val_count = len(os.listdir(os.path.join(self.splitted_data_path, "val", cls)))
                test_count = len(os.listdir(os.path.join(self.splitted_data_path, "test", cls))) if 'test' in self.splits else 0
                total_count = train_count + val_count + test_count

                train_str = f"train: {train_count}".ljust(15)
                val_str = f"val: {val_count}".ljust(15)
                test_str = f"test: {test_count}".ljust(15) if 'test' in self.splits else ''
                total_str = f"total: {total_count}".ljust(15)

                print(f'\t\t{cls.ljust(20)} -> {train_str} {val_str} {test_str} {total_str}')

            print('\n\n')
        return dataloaders, dataset_sizes, classes
    

    def prepare_set_ml(self, split):
        # Prepare data for ML models for a single split
        data_loader = self.dataloaders[split]
        features, labels = [], []
        for inputs, targets in data_loader:
            features.append(inputs.view(inputs.size(0), -1).numpy())  # Flatten and convert to NumPy
            labels.append(targets.numpy())
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels


    def _split(self):
        # check if the data folder is existed
        if not os.path.exists(self.images_folder_path):
            # self._download() # download zip data
            self._unzip() # unzip the data
        # split the data folder into train and val
        print('[INFO]  Splitting {} dataset...'.format(self.name))
        if not os.path.exists(self.images_folder_path):
            os.makedirs(self.images_folder_path)
        import splitfolders
        splitfolders.ratio(self.images_folder_path, output=self.splitted_data_path, ratio=self.split_ratio, seed=1998)
        print('')

    def _unzip(self):
        import zipfile
        with zipfile.ZipFile(os.path.join(self.download_path, self.zip_file), 'r') as file:
            file.extractall(self.unzip_path)

    def _count_datafiles(self):
        files = 0
        for _, dirnames, filenames in os.walk(self.splitted_data_path):
            files += len(filenames)
        return files

    def _clean(self):
        log_path = "files-scan-log.txt"
        removed = 0
        i = 0
        for dirname in os.listdir(self.splitted_data_path):
            current_set = os.path.join(self.splitted_data_path, dirname)
            for classname in os.listdir(current_set):
                current_dir = os.path.join(current_set, classname)
                for filename in os.listdir(current_dir):
                    i += 1
                    print('\r[INFO]  Scanning {}/{}'.format(i, self.length), end='')
                    path = os.path.join(current_dir, filename)
                    if os.path.getsize(path) == 0:
                        os.remove(path)
                        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                        with open(log_path, '+a') as f:
                            f.write('\nfile:{}\ttime:{}'.format(path, now))
                        removed += 1
        if removed == 0:
            print('\n[INFO]  No corruption')
        elif removed == 1:
            print('\n[INFO]  1 file has been removed. check {} for more details.'.format(log_path))
        else:
            print('\n[INFO]  {} files have been removed. check {} for more details.'.format(removed, log_path))
