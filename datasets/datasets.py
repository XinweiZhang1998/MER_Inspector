import torch, torchvision, torchtext
import random
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import(
    get_tokenizer,
    ngrams_iterator,
)
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torch.utils.data import Subset

import argparse
import torch.nn.functional as F
import requests
import tarfile
import PIL
from PIL import Image
from torchvision.datasets import ImageFolder
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import os
import pandas as pd
from torch.utils import data
import pickle
import sys
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import ImageFolder
import random
import os.path as osp
import numpy as np
from collections import defaultdict as dd
from PIL import Image
from ml_datasets import loaders

classes_dict = {
    "kmnist": 10,
    "mnist": 10,
    "cifar10": 10,
    "cifar10_gray": 10,
    "cifar100": 100,
    "svhn": 10,
    "gtsrb": 43,
    "fashionmnist": 10,
    "fashionmnist_32": 10,
    "mnist_32": 10,
    "ag_news":4,
    "stl10":10,
    "imagenette":10,
    "emnist":10,
    "artbench10":10,
    "voc2006":13,
    "celeba":8,
    
}

class ArtBench10(CIFAR10):
    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(ArtBench10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    def download(self):
            if self._check_integrity():
                print('Files already downloaded and verified')
                return
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

def get_nclasses(dataset: str):
    if dataset in classes_dict:
        return classes_dict[dataset]
    else:
        raise Exception("Invalid dataset")


# def yield_tokens(data_iter, ngrams):
#     for _, text in data_iter:
#         yield ngrams_iterator(tokenizer(text), ngrams)
# 准备数据加载器的函数
class AGNewsDataset(Dataset):
    def __init__(self, iterator, text_pipeline, label_pipeline, ngrams=2):
        self.samples = [(text_pipeline(text), label_pipeline(label)) for label, text in iterator]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def remove_unidentifiable_images(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img.verify()  # 验证图像
                except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                    print(f"无法识别的图像: {img_path}. 正在删除...")
                    os.remove(img_path)

def prepare_dataloaders(batch_size, num_workers):
    tokenizer = get_tokenizer("basic_english")
    def text_pipeline(x): return vocab(list(ngrams_iterator(tokenizer(x),ngrams=2)))
    def label_pipeline(x): return int(x) - 1

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets

    def yield_tokens(data_iter,ngrams=2):
        for _, text in data_iter:
            yield ngrams_iterator(tokenizer(text),ngrams)

    # Building the vocab
    #train_iter = DATASETS[dataset_name](root=root_dir, split='train')

    # Preparing the dataloaders
    train_iter, test_iter = torchtext.datasets.AG_NEWS(root='./data', split=('train', 'test'))
    #train_iter, test_iter = DATASETS[dataset_name](root=root_dir)
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    train_dataset = AGNewsDataset(train_iter, text_pipeline, label_pipeline, ngrams=2)
    test_dataset = AGNewsDataset(test_iter, text_pipeline, label_pipeline, ngrams=2)
    #num_train = int(len(train_dataset) * split_ratio)
    #split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=num_workers)
    #valid_loader = DataLoader(split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=num_workers)

    return train_loader, test_loader

def download_and_extract(url, destination_folder='./data'):
    os.makedirs(destination_folder, exist_ok=True)
    filename = os.path.join(destination_folder, "imagenette2-160.tgz")

    if not os.path.exists(filename):
        print("Downloading imagenette2-160 dataset...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
        print("Download completed.")
    else:
        print("Dataset already downloaded.")

    # 解压数据集
    print("Extracting dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=destination_folder)
    print("Extraction done.")

def download_and_extract_voc2006(url, destination_folder='./data'):
    os.makedirs(destination_folder, exist_ok=True)
    filename = os.path.join(destination_folder, "voc2006_trainval.tar")

    if not os.path.exists(filename):
        print("Downloading voc2006 dataset...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
        print("Download completed.")
    else:
        print("Dataset already downloaded.")

    # 解压数据集
    print("Extracting dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=destination_folder)
    print("Extraction done.")

class GTSRB(Dataset):
    base_folder = "GTSRB"

    def __init__(self, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = "./data"

        self.sub_directory = "trainingset" if train else "testset"
        self.csv_file_name = "training.csv" if train else "test.csv"

        csv_file_path = os.path.join(
            self.root_dir, self.base_folder, self.sub_directory, self.csv_file_name
        )

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root_dir,
            self.base_folder,
            self.sub_directory,
            self.csv_data.iloc[idx, 0],
        )
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


def get_dataset(dataset, batch_size=256, augment=False):
    if dataset in ["mnist", "kmnist", "fashionmnist", "cifar10", "cifar100","svhn","gtsrb","stl10","imagenette","emnist","artbench10","voc2006","celeba"]:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        num_workers = 4
        if dataset in ["mnist", "kmnist", "fashionmnist"]:
            if augment:
                transform_train = transforms.Compose(
                    [
                        transforms.RandomCrop(28, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    ]
                )
            else:
                transform_train = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
                )
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            )
        elif dataset in ["mnist_32", "fashionmnist_32"]:
            transform_train = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        elif dataset in ["svhn"]:
            transform_train = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        elif dataset in ["stl10"]:
            transform_train = transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            transform_test = transforms.Compose(
                [   
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        elif dataset in ["imagenette"]:
            transform_train = transforms.Compose(
                [   
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            transform_test = transforms.Compose(
                [   
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        elif dataset in ["emnist"]:
            transform_train = transforms.Compose(
                [   
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            transform_test = transforms.Compose(
                [   
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        elif dataset in ["artbench10"]:
            transform_train = transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomRotation(15),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    # transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            transform_test = transforms.Compose(
                [   
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        
        elif dataset in ["gtsrb"]:
            transform_train = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        elif dataset in ["cifar10_gray"]:
            if augment:
                transform_train = transforms.Compose(
                    [
                        transforms.Grayscale(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    ]
                )
            else:
                transform_train = transforms.Compose(
                    [
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    ]
                )

            transform_test = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )

        else:
            if augment:
                transform_train = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
            else:
                transform_train = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std),]
                )

            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )

        if dataset in ["mnist", "mnist_32"]:
            trainset = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.MNIST(
                root="./data", train=False, download=True, transform=transform_test
            )
        elif dataset in ["kmnist"]:
            trainset = torchvision.datasets.KMNIST(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.KMNIST(
                root="./data", train=False, download=True, transform=transform_test
            )

        elif dataset in ["fashionmnist", "fashionmnist_32"]:
            trainset = torchvision.datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform_test
            )

        elif dataset in ["cifar10", "cifar10_gray"]:
            trainset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform_train
            )
            #(50000, 3, 32, 32)
            # transform is the preprocessing process for dataset.
            testset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform_test
            )

        elif dataset == "cifar100":
            trainset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform_test
            )

        elif dataset == "svhn":
            trainset = torchvision.datasets.SVHN(
                root="./data", split="train", download=True, transform=transform_train
            )
            testset = torchvision.datasets.SVHN(
                root="./data", split="test", download=True, transform=transform_test
            )

        elif dataset == "stl10":
            trainset = torchvision.datasets.STL10(
                root="./data", split="train", download=True, transform=transform_train
            )
            # train_image, train_label = trainset[0]
            # # Print out the size of the training image and the training set size
            # print(f"Training Set Size: {len(trainset)}, Image Dimensions: {train_image.size()}","Training Label Dimensions",train_label)
            testset = torchvision.datasets.STL10(
                root="./data", split="test", download=True, transform=transform_test
            )
            
        elif dataset == "celeba":
            class CustomCelebADataset(Dataset):
                def __init__(self, image_dir, attr_file, split_file, split, transform=None):
                    # 图像目录
                    self.image_dir = image_dir
                    # 加载属性
                    self.attr_df = pd.read_csv(attr_file, sep="\s+", skiprows=1,index_col=0)
                    # self.attr_df = pd.read_csv(attr_file, sep="\s+", header=1)
                    # 加载分割
                    self.split_df = pd.read_csv(split_file, sep="\s+", header=None, names=['image_id', 'split'])
                    # 过滤所需的分割
                    self.split = split
                    self.image_files = self.split_df[self.split_df['split'] == split]['image_id'].values
                    self.transform = transform

                def __len__(self):
                    return len(self.image_files)

                def __getitem__(self, idx):
                    image_name = self.image_files[idx]
                    image_path = os.path.join(self.image_dir, image_name)
                    image = Image.open(image_path)

                    if self.transform:
                        image = self.transform(image)

                    # Here we select only the three specified attributes
                    attributes = ['Heavy_Makeup', 'Mouth_Slightly_Open', 'Smiling']
                    labels = self.attr_df.loc[image_name, attributes].values.astype(int)
                    # Convert attribute presence (1 or -1 in the dataset) to a binary (0 or 1)
                    labels = (labels + 1) // 2
                    # Combine the three attributes to form a single label for 8-class classification
                    class_label = labels.dot(1 << np.arange(labels.size)[::-1])

                    return image, torch.tensor(class_label, dtype=torch.long)

            # 定义转换
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # 添加其他所需转换
            ])

            # 创建数据集实例
            trainset = CustomCelebADataset(
                image_dir='./data/celeba/img_align_celeba',
                attr_file='./data/celeba/list_attr_celeba.txt',
                split_file='./data/celeba/list_eval_partition.txt',
                split=0,  # 0表示训练集
                transform=transform
            )

            testset = CustomCelebADataset(
                image_dir='./data/celeba/img_align_celeba',
                attr_file='./data/celeba/list_attr_celeba.txt',
                split_file='./data/celeba/list_eval_partition.txt',
                split=2,  # 2表示测试集
                transform=transform
            )
            def create_subset(dataset, fraction):
                # 计算子集大小
                subset_size = int(len(dataset) * fraction)
                # 生成随机索引
                np.random.seed(42)
                indices = np.random.choice(len(dataset), subset_size, replace=False)
                # 创建子集
                return Subset(dataset, indices)
            trainset = create_subset(trainset, 0.2)
            testset = create_subset(testset, 0.2)
            train_image, train_label = trainset[0]
            # Print out the size of the training image and the training set size
            print(f"Training Set Size: {len(trainset)}, Image Dimensions: {train_image.size()}","Training Label Dimensions",train_label)

            # Get a single image and its label from the test set
            test_image, test_label = testset[0]
            # Print out the size of the test image and the test set size
            print(f"Test Set Size: {len(testset)}, Image Dimensions: {test_image.size()}",f"Test Label Dimensions: {test_label.shape}")
            

        elif dataset == "imagenette":
            def dataset_exists(path):
                return os.path.exists(path)

            dataset_root = './data/imagenette2-160'
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'

            if not dataset_exists(dataset_root):
                # 如果数据集不存在，下载并提取
                download_and_extract(url)

            trainset = ImageFolder(root='/root/autodl-tmp/project/data/imagenette2-160/train', transform=transform_train)
            testset = ImageFolder(root='/root/autodl-tmp/project/data/imagenette2-160/val', transform=transform_test)
        
        elif dataset == "voc2006":
            def dataset_exists(path):
                return os.path.exists(path)

            dataset_root = './data/voc2006'
            url = 'http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar'

            if not dataset_exists(dataset_root):
                # 如果数据集不存在，下载并提取
                download_and_extract_voc2006(url)

            trainset = ImageFolder(root='/root/autodl-tmp/project/data/voc2006_trainval/train', transform=transform_train)
            testset = ImageFolder(root='/root/autodl-tmp/project/data/voc2006_trainval/val', transform=transform_test)

        elif dataset == "artbench10":
            def dataset_exists(path):
                return os.path.exists(path)

            trainset = ArtBench10(root='./data', train=True, download=True, transform=transform_train)
            testset = ArtBench10(root='./data', train=False, download=True, transform=transform_test)
            # 计算训练集和测试集的长度
            trainset_length = len(trainset)
            testset_length = len(testset)

            # 打印长度
            print(f"训练集长度: {trainset_length}")
            print(f"测试集长度: {testset_length}")

        elif dataset == "emnist":
            trainset = torchvision.datasets.EMNIST(
                root="./data",
                split='digits',  # You can change the split based on your needs (e.g., 'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
                train=True,
                download=True,
                transform=transform_train
            )
            testset = torchvision.datasets.EMNIST(
                root="./data",
                split='digits',  # Make sure this matches the train split
                train=False,
                download=True,
                transform=transform_test
            )


        elif dataset == "gtsrb":
            trainset = torchvision.datasets.GTSRB(
                root="./data", split="train", download=True, transform=transform_train
            )
            testset = torchvision.datasets.GTSRB(
                root="./data", split="test", download=True, transform=transform_test
            )
            # trainset = GTSRB(train=True, transform=transform_train)
            # testset = GTSRB(train=False, transform=transform_test)

        else:
            sys.exit("Unknown dataset {}".format(dataset))

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True, #打乱
            num_workers=num_workers,
            pin_memory=True,
        )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    if dataset in ["ag_news"]:
        num_workers = 4
        tokenizer = get_tokenizer('basic_english')
        train_iter = torchtext.datasets.AG_NEWS(root='./data',split='train')
        # 分词生成器
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        # 构建词汇表
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        # 设置默认索引，当某个单词不在词汇表中，则返回0
        vocab.set_default_index(vocab["<unk>"])
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(len(vocab))
        
        def collate_batch(batch):
            label_list, text_list = [], []
            max_length = 100  # 设置最大长度为 100
            for (_label, _text) in batch:
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
                # 截断或填充处理
                if len(processed_text) > max_length:
                    processed_text = processed_text[:max_length]  # 截断超过 100 的部分
                elif len(processed_text) < max_length:
                    processed_text = F.pad(processed_text, (0, max_length - len(processed_text)), value=1)  # 使用 1 填充
                text_list.append(processed_text)
            # 填充文本列表使其具有相同的长度
            text_list = pad_sequence(text_list, batch_first=True, padding_value=1)
            return  text_list.to(device), torch.tensor(label_list, dtype=torch.int64).to(device)

        train_iter = torchtext.datasets.AG_NEWS(root='./data',split='train')
        trainloader = torch.utils.data.DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        test_iter = torchtext.datasets.AG_NEWS(root='./data',split='test')
        testloader = torch.utils.data.DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # for data, target in trainloader:
        #     print("Data shape:", data.shape)   # This will print the shape of the data tensor
        #     print("Target shape:", target.shape)  # This will print the shape of the target tensor
        #     print(data)
        #     print(target)
        #     break
 
        
    
    return trainloader, testloader
