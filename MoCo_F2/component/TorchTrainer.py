import os.path
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
from torch import optim
from PIL import Image
import torch.nn as nn
# import file_paths
import sys
from importlib import import_module

# sys.path.append('D:/PythonProjects/pytorch_v2/seed_models')


class File_Paths:
    def __init__(self):
        self.DATASET_PATH = "D:/SaveProjects/MoCo(Save)/codes/datasets(light)"


file_paths = File_Paths()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


IMAGE_NUM = 100
BATCH_SIZE = 5
test_loader_28_mn = ""
test_loader_224_im = ""
test_loader_244_im = ""
test_loader_299_im = ""
test_loader_224_cf = ""
test_loader_sp = ""


def get_mnist() -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'MNIST.npz'))
    images = data['x_train']
    labels = data['y_train']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = MyDataset(images, labels, transform=transform)
    sub_dataset = torch.utils.data.Subset(dataset=dataset, indices=range(IMAGE_NUM))
    dataloader = DataLoader(sub_dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


def get_mnist_test() -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'MNIST.npz'))
    images = data['x_test']
    labels = data['y_test']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = MyDataset(images, labels, transform=transform)
    sub_dataset = torch.utils.data.Subset(dataset=dataset, indices=range(IMAGE_NUM))
    dataloader = DataLoader(sub_dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


def get_imagenet(size: int = 224) -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'imagenet.npz'))
    images = data['x_test']
    labels = data['y_test']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if size == 224:
        dataset = MyDataset(images[:IMAGE_NUM], labels[:IMAGE_NUM], transform=transform)
    elif size == 244:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((244, 244))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        dataset = MyDataset(resized_images, labels[:IMAGE_NUM], transform=transform)
    elif size == 299:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((299, 299))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        dataset = MyDataset(resized_images, labels[:IMAGE_NUM], transform=transform)
    else:
        dataset = MyDataset(images[:IMAGE_NUM], labels[:IMAGE_NUM], transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


def get_imagenet_test(size: int = 224) -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'imagenet.npz'))
    images = data['x_test']
    labels = data['y_test']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if size == 224:
        dataset = MyDataset(images[IMAGE_NUM: IMAGE_NUM * 2], labels[IMAGE_NUM: IMAGE_NUM * 2], transform=transform)
    elif size == 244:
        resized_images = []
        for i in range(IMAGE_NUM, IMAGE_NUM * 2):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((244, 244))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        dataset = MyDataset(resized_images, labels[IMAGE_NUM: IMAGE_NUM * 2], transform=transform)
    elif size == 299:
        resized_images = []
        for i in range(IMAGE_NUM, IMAGE_NUM * 2):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((299, 299))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        dataset = MyDataset(resized_images, labels[IMAGE_NUM: IMAGE_NUM * 2], transform=transform)
    else:
        dataset = MyDataset(images[IMAGE_NUM: IMAGE_NUM * 2], labels[IMAGE_NUM: IMAGE_NUM * 2], transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


def get_cifar10() -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'cifar10.npz'))
    images = data['x_train']
    labels = data['y_train']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    resized_images = []
    for i in range(IMAGE_NUM):
        image = images[i]
        resized_image = image.astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((224, 224))
        resized_image = np.array(resized_image)
        resized_images.append(resized_image)
    dataset = MyDataset(resized_images, labels[:IMAGE_NUM], transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


def get_cifar10_test() -> DataLoader:
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'cifar10.npz'))
    images = data['x_train']
    labels = data['y_train']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    resized_images = []
    for i in range(IMAGE_NUM, IMAGE_NUM * 2):
        image = images[i]
        resized_image = image.astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((224, 224))
        resized_image = np.array(resized_image)
        resized_images.append(resized_image)
    dataset = MyDataset(resized_images, labels[IMAGE_NUM: IMAGE_NUM * 2], transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del data, images, labels, dataset
    return dataloader


input_size = 1  # 输入维度
hidden_size = 64  # LSTM隐藏层维度
output_size = 1  # 输出维度
lr = 0.001  # 学习率


class StockPriceDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx]


def get_stockprice() -> DataLoader:
    data = pd.read_csv(os.path.join(file_paths.DATASET_PATH, 'DIS.csv'))
    price_sequence = data.iloc[:, 1].values.astype(np.float32)
    price_sequence = price_sequence[:100]
    dataset = StockPriceDataset(price_sequence)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


def get_stockprice_test() -> DataLoader:
    data = pd.read_csv(os.path.join(file_paths.DATASET_PATH, 'DIS.csv'))
    price_sequence = data.iloc[:, 1].values.astype(np.float32)
    price_sequence = price_sequence[100:200]
    dataset = StockPriceDataset(price_sequence)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


def get_dataloader(net_name: str, image_num: int, batch_size: int) -> DataLoader[Any] | None:  # net name as 'lenet'...
    image_num = image_num
    batch_size = batch_size
    if net_name == 'lenet':  # MNIST dataloader
        return get_mnist()

    elif net_name in ['ResNet18', 'ResNet50', 'mobilenet', 'densenet',  # ImageNet dataloader
                      'xception', 'googlenet']:
        return get_imagenet(224)

    elif net_name == 'InceptionV3':
        return get_imagenet(299)

    elif net_name in ['squeezenet', 'nasnet']:
        return get_imagenet(244)

    elif net_name in ['vgg16', 'vgg19', 'alexnet']:  # CIFAR-10 dataloader
        return get_cifar10()

    elif net_name in ['BiLSTM', 'LSTM', 'GRU']:  # Stock-Price dataloader
        # TODO: Stock-Price
        return None

    else:
        return None


def reshape_tensor(input_tensor, target_shape: tuple):
    assert len(target_shape) == 2
    batch_size = target_shape[0]
    length = target_shape[1]
    if len(input_tensor.shape) == 0 or len(input_tensor.shape) == 1:
        output_tensor = torch.randn(target_shape)
    else:
        temp = input_tensor.flatten(1, -1)
        # fix size 0 (batch size)
        if temp.size(0) > batch_size:
            temp = temp[:batch_size, :]
        elif temp.size(0) == batch_size:
            temp = temp
        else:
            temp = torch.cat((temp, torch.zeros(batch_size - temp.size(0), temp.size(1)).to('cuda')), dim=0)

        # fix size 1 (label num)
        if temp.size(1) > length:
            temp = temp[:, :length]
        elif temp.size(1) == length:
            temp = temp
        else:
            temp = torch.cat((temp, torch.zeros(temp.size(0), length - temp.size(1)).to('cuda')), dim=1)

        # fix finished, now shape is target_shape
        output_tensor = temp

    return output_tensor


class Trainer:
    def __init__(self):
        MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
                      'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                      'densenet', 'LSTM', 'GRU', 'googlenet', 'BiLSTM']
        self.dataloader_dict = {}
        self.train_count = 0
        self.dataloader_dict['resnet18'] = get_imagenet(224)
        self.dataloader_dict['alexnet'] = get_cifar10()
        self.dataloader_dict['LeNet'] = get_mnist()
        self.dataloader_dict['mobilenet'] = self.dataloader_dict['resnet18']
        self.dataloader_dict['squeezenet'] = self.dataloader_dict['resnet18']
        self.dataloader_dict['vgg16'] = self.dataloader_dict['alexnet']
        self.dataloader_dict['vgg19'] = self.dataloader_dict['alexnet']
        self.dataloader_dict['lstm'] = get_stockprice()
        self.dataloader_dict['googlenet'] = self.dataloader_dict['resnet18']
        return

    def train(self, net_to_train: nn.Module, net_name: str) -> None:
        # pointnet
        if net_name == "pointnet":
            net = net_to_train
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            for i in range(20):
                reshaped_inputs = torch.randn(BATCH_SIZE, 3, 5)
                labels = torch.randn(BATCH_SIZE, 10)
                optimizer.zero_grad()  # 梯度清零

                outputs = net(reshaped_inputs)  # 前向传播
                outputs = reshape_tensor(outputs, (BATCH_SIZE, 10))
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
            return
        # pointnet

        if net_name in self.dataloader_dict.keys():
            dataloader = self.dataloader_dict[net_name]
        else:
            dataloader = None
        if dataloader is None:
            print(str(net_to_train).split('(', 1)[0] + ' no train')
            return
        net = net_to_train

        # rnn
        if net_name in ['lstm', 'BiLSTM', 'GRU']:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)
            for i, inputs in enumerate(dataloader):
                reshaped_inputs = torch.unsqueeze(inputs, dim=1)
                labels = inputs.clone()  # 在这个示例中，将输入作为标签（仅用于示例目的，您可能需要更改为目标变量）

                optimizer.zero_grad()  # 梯度清零

                outputs = net(reshaped_inputs)  # 前向传播
                outputs = reshape_tensor(outputs, (5, 1))
                labels = reshape_tensor(labels, (5, 1))
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
            # print('Train finished (RNN)')
            return

        net.to('cuda')
        criterion = nn.CrossEntropyLoss()
        criterion.to('cuda')
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for param in optimizer.param_groups:
            for key, value in param.items():
                if isinstance(value, torch.Tensor):
                    param[key] = value.to('cuda')
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # forward:
            inputs, labels = data
            labels = labels.to('cuda')
            inputs = inputs.to('cuda')
            optimizer.zero_grad()
            outputs = net(inputs).to('cuda')
            if net_name in ['alexnet', 'vgg16', 'vgg19']:
                labels = labels.squeeze()
                labels = labels.to('cuda')

            # outputs reshape:
            if net_name in ['lenet', 'alexnet', 'vgg16', 'vgg19']:
                target_shape = (BATCH_SIZE, 10)
            else:
                target_shape = (BATCH_SIZE, 1000)
            outputs = reshape_tensor(outputs, target_shape)
            outputs = outputs.to('cuda')

            # loss calculation and backward:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(str(i) + ':  ' + str(loss.item()))
        self.train_count += 1
        # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
        # torch.cpu.empty_cache()
        return


# class Tester:
#     def __init__(self):
#         MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
#                       'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
#                       'densenet', 'LSTM', 'GRU', 'googlenet', 'BiLSTM']
#         self.dataloader_dict = {}
#         self.train_count = 0
#         self.dataloader_dict['ResNet18'] = get_imagenet_test(224)
#         self.dataloader_dict['ResNet50'] = self.dataloader_dict['ResNet18']
#         self.dataloader_dict['nasnet'] = get_imagenet_test(244)
#         self.dataloader_dict['InceptionV3'] = get_imagenet_test(299)
#         self.dataloader_dict['xception'] = self.dataloader_dict['ResNet18']
#         self.dataloader_dict['alexnet'] = get_cifar10_test()
#         self.dataloader_dict['lenet'] = get_mnist_test()
#         self.dataloader_dict['mobilenet'] = self.dataloader_dict['ResNet18']
#         self.dataloader_dict['squeezenet'] = self.dataloader_dict['nasnet']
#         self.dataloader_dict['vgg16'] = self.dataloader_dict['alexnet']
#         self.dataloader_dict['vgg19'] = self.dataloader_dict['alexnet']
#         self.dataloader_dict['densenet'] = self.dataloader_dict['ResNet18']
#         self.dataloader_dict['LSTM'] = get_stockprice_test()
#         self.dataloader_dict['GRU'] = self.dataloader_dict['LSTM']
#         self.dataloader_dict['googlenet'] = self.dataloader_dict['ResNet18']
#         self.dataloader_dict['BiLSTM'] = self.dataloader_dict['LSTM']
#         return
#
#     def test(self, net_to_train: nn.Module, net_name: str) -> float | None:
#         if net_name in self.dataloader_dict.keys():
#             dataloader = self.dataloader_dict[net_name]
#         else:
#             dataloader = None
#         if dataloader is None:
#             print(str(net_to_train).split('(', 1)[0] + ' no train')
#             return
#         net = net_to_train
#
#         # rnn
#         if net_name in ['LSTM', 'BiLSTM', 'GRU']:
#             correct = 0
#             total = 0
#             for i, inputs in enumerate(dataloader):
#                 reshaped_inputs = torch.unsqueeze(torch.unsqueeze(inputs, dim=1), dim=2)
#                 labels = inputs.clone()  # 在这个示例中，将输入作为标签（仅用于示例目的，您可能需要更改为目标变量）
#
#                 outputs = net(reshaped_inputs)  # 前向传播
#                 outputs = reshape_tensor(outputs, (5, 1))
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#             # print('Train finished (RNN)')
#             return correct / total
#
#         net.to('cuda')
#         criterion = nn.CrossEntropyLoss()
#         criterion.to('cuda')
#         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#         for param in optimizer.param_groups:
#             for key, value in param.items():
#                 if isinstance(value, torch.Tensor):
#                     param[key] = value.to('cuda')
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for i, data in enumerate(dataloader, 0):
#             # forward:
#             inputs, labels = data
#             labels = labels.to('cuda')
#             inputs = inputs.to('cuda')
#             outputs = net(inputs).to('cuda')
#             if net_name in ['alexnet', 'vgg16', 'vgg19']:
#                 labels = labels.squeeze()
#                 labels = labels.to('cuda')
#
#             # outputs reshape:
#             if net_name in ['lenet', 'alexnet', 'vgg16', 'vgg19']:
#                 target_shape = (BATCH_SIZE, 10)
#             else:
#                 target_shape = (BATCH_SIZE, 1000)
#             outputs = reshape_tensor(outputs, target_shape)
#             outputs = outputs.to('cuda')
#
#             # loss calculation and backward:
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # print(str(i) + ':  ' + str(loss.item()))
#         self.train_count += 1
#         # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
#         # torch.cpu.empty_cache()
#         return correct / total


torch_trainer = Trainer()
