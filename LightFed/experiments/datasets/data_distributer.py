import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets as vision_datasets
from lightfed.tools.funcs import save_pkl


class TransDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.dataset)


class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class DataDistributer:
    def __init__(self, args, dataset_dir=None, cache_dir=None):
        if dataset_dir is None:
            dataset_dir = os.path.abspath(os.path.join(__file__, "../../../../dataset"))

        if cache_dir is None:
            cache_dir = f"{dataset_dir}/cache_data"

        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.args = args
        self.client_num = args.client_num
        self.batch_size = args.batch_size

        self.class_num = None
        self.x_shape = None  
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        self.train_dataloaders = None
        self.test_dataloaders = None

        # _cache_file_name = f"{self.cache_dir}/{self.args.data_set}_seed_{self.args.seed}_client_num_{self.client_num}_{self.args.data_partition_mode}"
        # if self.args.data_partition_mode == 'non_iid_dirichlet':
        #     _cache_file_name += f"_{self.args.non_iid_alpha}"
        # _cache_file_name += f"{self.args.device.type}.pkl"

        # if os.path.exists(_cache_file_name):
        #     self.class_num, self.x_shape, \
        #         self.client_train_dataloaders, self.client_test_dataloaders, \
        #         self.train_dataloaders, self.test_dataloaders = load_pkl(_cache_file_name)
        #     return

 
        _dataset_load_func = getattr(self, f'_load_{args.data_set.replace("-","_")}')
        _dataset_load_func()

        # save_pkl((self.class_num, self.x_shape,
        #           self.client_train_dataloaders, self.client_test_dataloaders,
        #           self.train_dataloaders, self.test_dataloaders),
        #          _cache_file_name)

    def get_client_train_dataloader(self, client_id):
        return self.client_train_dataloaders[client_id]

    def get_client_test_dataloader(self, client_id):
        return self.client_test_dataloaders[client_id]

    def get_train_dataloader(self):
        return self.train_dataloaders

    def get_test_dataloader(self):
        return self.test_dataloaders

    def _load_MNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=True, download=True, transform=transform)
        test_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=False, download=True, transform=transform)
        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=True)

        if self.args.data_partition_mode == 'None':
            return

        # client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        client_train_datasets = self._split_dataset(train_dataset, test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            # _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=True)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)

                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)
                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)
            self.client_train_dataloaders.append(_train_dataloader)
            # self.client_test_dataloaders.append(_test_dataloader)

    def _load_EMNIST(self):
        self.class_num = 62
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="byclass", train=True, download=True, transform=transform)
        test_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="byclass", train=False, download=True, transform=transform)

        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=True)

        if self.args.data_partition_mode == 'None':
            return

        # client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        client_train_datasets = self._split_dataset(train_dataset, test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            # _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=True)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)

                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)
                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_train_dataset), shuffle=True)
            self.client_train_dataloaders.append(_train_dataloader)
            # self.client_test_dataloaders.append(_test_dataloader)


    def _load_CIFAR_10(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)


        train_transform = transforms.Compose([
            # transforms.RandomRotation(degrees=10),  
            transforms.RandomCrop(size=32, padding=4),  
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ToTensor(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

        raw_train_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=True, download=True)
        raw_test_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=False, download=True)

        train_dataset = TransDataset(raw_train_dataset, train_transform)
        test_dataset = TransDataset(raw_test_dataset, test_transform)
        ###train data
        if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=True)

        if self.args.data_partition_mode == 'None':
            return

        # raw_client_train_datasets, raw_client_test_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        raw_client_train_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _raw_train_dataset = raw_client_train_datasets[client_id]
            # _raw_test_dataset = raw_client_test_datasets[client_id]
            _train_dataset = TransDataset(_raw_train_dataset, train_transform)
            # _test_dataset = TransDataset(_raw_test_dataset, test_transform)
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=True)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)

                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)
                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            self.client_train_dataloaders.append(_train_dataloader)
            # self.client_test_dataloaders.append(_test_dataloader)

    def _load_CIFAR_100(self):
        self.class_num = 100
        self.x_shape = (3, 32, 32)

        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        raw_train_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=True, download=True)
        raw_test_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=False, download=True)

        train_dataset = TransDataset(raw_train_dataset, train_transform)
        test_dataset = TransDataset(raw_test_dataset, test_transform)
        ###train data
        if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        #test data
        if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)


        if self.args.data_partition_mode == 'None':
            return

        # raw_client_train_datasets, raw_client_test_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _raw_train_dataset = raw_client_train_datasets[client_id]
            # _raw_test_dataset = raw_client_test_datasets[client_id]
            _train_dataset = TransDataset(_raw_train_dataset, train_transform)
            # _test_dataset = TransDataset(_raw_test_dataset, test_transform)
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=True)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)

                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=True)
                # _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=True)

            self.client_train_dataloaders.append(_train_dataloader)
            # self.client_test_dataloaders.append(_test_dataloader)

    def _split_dataset(self, train_dataset, test_dataset):

        if self.args.data_partition_mode == 'iid':
            partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1/self.client_num)
        elif self.args.data_partition_mode == 'non_iid_dirichlet':
            partition_proportions = np.random.dirichlet(alpha=np.full(shape=self.client_num, fill_value=self.args.non_iid_alpha), size=self.class_num)
        else:
            raise Exception(f"unknow data_partition_mode:{self.args.data_partition_mode}")

        client_train_datasets = self._split_dataset_by_proportion(train_dataset, partition_proportions)
        # client_test_datasets = self._split_dataset_by_proportion(test_dataset, partition_proportions)
        return client_train_datasets

    def _split_dataset_by_proportion(self, dataset, partition_proportions):
        data_labels = dataset.targets

        class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                      for y in range(self.class_num)]


        client_idcs = [[] for _ in range(self.client_num)]

        if self.args.data_partition_mode == 'non_iid_dirichlet':
            for _ in range(2):
                for client_id in range(self.client_num):  # 每个客户端至少一个样本
                    class_id = np.random.randint(low=0, high=self.class_num)
                    client_idcs[client_id].append(class_idcs[class_id].pop())

        for c, fracs in zip(class_idcs, partition_proportions):

            np.random.shuffle(c)
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i].extend(list(idcs))

        client_data_list = [[] for _ in range(self.client_num)]
        for client_id, client_data in zip(client_idcs, client_data_list):
            for id in client_id:
                client_data.append(dataset[id])
        # label_partition = []
        # for dtset in client_data_list:
        #     label_partition.append([lb[1] for lb in dtset])
        #
        # save_pkl(label_partition, file_path=f"{os.path.dirname(__file__)}/label_partition/{self.args.data_set}_{self.args.non_iid_alpha}")
        client_datasets = []
        for client_data in client_data_list:

            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets
