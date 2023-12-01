import os.path
import jittor
import pandas as pd
from jittor.transform import ImageNormalize
# from torch.utils.data import Dataset, DataLoader
# import torch
import numpy as np
# from torchvision import transforms
from jittor import optim
from PIL import Image
import jittor.nn as nn
# import file_paths
import sys
from importlib import import_module
import jittor_utils

# sys.path.append('D:/PythonProjects/jittor_v2_0709_b/seed_models')
IMAGE_NUM = 100
BATCH_SIZE = 5
# os.environ['TRAIN_STOP_FLAG'] = '0'


class File_Paths:
    def __init__(self):
        self.DATASET_PATH = "D:/SaveProjects/MoCo(Save)/codes/datasets(light)"


file_paths = File_Paths()


# these getters will return (images, labels)
def get_mnist():
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'MNIST.npz'))
    images = data['x_train'][:IMAGE_NUM, :, :]
    images = np.expand_dims(images, axis=1)
    labels = data['y_train'][:IMAGE_NUM]
    return jittor.array(images / 255.0), jittor.array(labels / 10.0)


def get_mnist_test():
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'MNIST.npz'))
    images = data['x_test'][:IMAGE_NUM, :, :]
    images = np.expand_dims(images, axis=1)
    labels = data['y_test'][:IMAGE_NUM]
    return jittor.array(images / 255.0), jittor.array(labels / 10.0)


def get_imagenet(size: int = 224):
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'imagenet.npz'))
    images = data['x_test'][:IMAGE_NUM, :, :, :]
    labels = data['y_test'][:IMAGE_NUM]
    if size == 224:
        images = np.transpose(images, (0, 3, 1, 2))
        return jittor.array(images / 255.0), jittor.array(labels / 1000.0)
    elif size == 244:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((244, 244))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2))
        return jittor.array(resized_images / 255.0), jittor.array(labels / 1000.0)
    elif size == 299:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((299, 299))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2))
        return jittor.array(resized_images / 255.0), jittor.array(labels / 1000.0)
    else:
        return jittor.array(images / 255.0), jittor.array(labels / 1000.0)


def get_imagenet_test(size: int = 224):
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'imagenet.npz'))
    images = data['x_test'][IMAGE_NUM: IMAGE_NUM * 2, :, :, :]
    labels = data['y_test'][IMAGE_NUM: IMAGE_NUM * 2]
    if size == 224:
        images = np.transpose(images, (0, 3, 1, 2))
        return jittor.array(images / 255.0), jittor.array(labels / 1000.0)
    elif size == 244:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((244, 244))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2))
        return jittor.array(resized_images / 255.0), jittor.array(labels / 1000.0)
    elif size == 299:
        resized_images = []
        for i in range(IMAGE_NUM):
            image = images[i]
            resized_image = image.astype(np.uint8)
            resized_image = Image.fromarray(resized_image)
            resized_image = resized_image.resize((299, 299))
            resized_image = np.array(resized_image)
            resized_images.append(resized_image)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2))
        return jittor.array(resized_images / 255.0), jittor.array(labels / 1000.0)
    else:
        return jittor.array(images / 255.0), jittor.array(labels / 1000.0)


def get_cifar10():
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'cifar10.npz'))
    images = data['x_train'][:IMAGE_NUM, :, :, :]
    labels = data['y_train'][:IMAGE_NUM, :]
    resized_images = []
    for i in range(IMAGE_NUM):
        image = images[i]
        resized_image = image.astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((224, 224))
        resized_image = np.array(resized_image)
        resized_images.append(resized_image)
    resized_images = np.transpose(resized_images, (0, 3, 1, 2))
    return jittor.array(resized_images / 255.0), jittor.array(labels / 10.0).squeeze()


def get_cifar10_test():
    data = np.load(os.path.join(file_paths.DATASET_PATH, 'cifar10.npz'))
    images = data['x_train'][IMAGE_NUM: IMAGE_NUM * 2, :, :, :]
    labels = data['y_train'][IMAGE_NUM: IMAGE_NUM * 2, :]
    resized_images = []
    for i in range(IMAGE_NUM):
        image = images[i]
        resized_image = image.astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((224, 224))
        resized_image = np.array(resized_image)
        resized_images.append(resized_image)
    resized_images = np.transpose(resized_images, (0, 3, 1, 2))
    return jittor.array(resized_images / 255.0), jittor.array(labels / 10.0).squeeze()


def get_stock_price():
    data = pd.read_csv(os.path.join(file_paths.DATASET_PATH, 'DIS.csv'))
    price_sequence = data.iloc[:, 1].values.astype(np.float32)
    price_sequence = price_sequence[:100]
    return jittor.array(price_sequence), jittor.array(price_sequence)


def get_stock_price_test():
    data = pd.read_csv(os.path.join(file_paths.DATASET_PATH, 'DIS.csv'))
    price_sequence = data.iloc[:, 1].values.astype(np.float32)
    price_sequence = price_sequence[100:200]
    return jittor.array(price_sequence), jittor.array(price_sequence)


def reshape_tensor(input_tensor: jittor.Var, target_shape: tuple):  # both accept and output is jittor Var
    assert len(target_shape) == 2
    batch_size = target_shape[0]
    length = target_shape[1]
    if len(input_tensor.shape) == 0 or len(input_tensor.shape) == 1:
        output_tensor = jittor.randn(target_shape)
    else:
        temp = input_tensor.flatten(1, -1)
        # fix size 0 (batch size)
        if temp.size(0) > batch_size:
            temp = temp[:batch_size, :]
        elif temp.size(0) == batch_size:
            temp = temp
        else:
            temp = jittor.cat((temp, jittor.zeros(batch_size - temp.size(0), temp.size(1))), dim=0)

        # fix size 1 (label num)
        if temp.size(1) > length:
            temp = temp[:, :length]
        elif temp.size(1) == length:
            temp = temp
        else:
            temp = jittor.cat((temp, jittor.zeros(temp.size(0), length - temp.size(1))), dim=1)

        # fix finished, now shape is target_shape
        output_tensor = temp

    return output_tensor


class Trainer:
    def __init__(self):
        MODEL_LIST = ['ResNet18', 'ResNet50', 'nasnet', 'InceptionV3', 'xception', 'testnet',
                      'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
                      'densenet', 'LSTM', 'GRU', 'BiLSTM', 'googlenet']
        self.dataloader_dict = {}
        self.train_count = 0
        self.dataloader_dict['resnet18'] = get_imagenet(224)
        self.dataloader_dict['alexnet'] = get_cifar10()
        self.dataloader_dict['LeNet'] = get_mnist()
        self.dataloader_dict['mobilenet'] = self.dataloader_dict['resnet18']
        self.dataloader_dict['squeezenet'] = self.dataloader_dict['resnet18']
        self.dataloader_dict['vgg16'] = self.dataloader_dict['alexnet']
        self.dataloader_dict['vgg19'] = self.dataloader_dict['alexnet']
        self.dataloader_dict['lstm'] = get_stock_price()
        self.dataloader_dict['googlenet'] = self.dataloader_dict['resnet18']
        return

    def train(self, net_to_train: nn.Module, net_name: str) -> None:
        # pointnet
        if net_name == "pointnet":
            net = net_to_train
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            for i in range(20):
                reshaped_inputs = jittor.randn(BATCH_SIZE, 3, 5)
                labels = jittor.randn(BATCH_SIZE, 10)
                optimizer.zero_grad()  # 梯度清零

                outputs = net(reshaped_inputs)  # 前向传播
                outputs = reshape_tensor(outputs, (BATCH_SIZE, 10))
                loss = criterion(outputs, labels)  # 计算损失
                optimizer.backward(loss)  # 反向传播
                optimizer.step()  # 更新参数
            return
        # pointnet

        if net_name in self.dataloader_dict.keys():
            dataloader = self.dataloader_dict[net_name]  # now, data_loader is a tuple: (images, labels)
        else:
            dataloader = None
        if dataloader is None:
            # print(str(net_to_train).split('(', 1)[0] + ' no train')
            return
        net = net_to_train
        images = dataloader[0]
        results = dataloader[1]
        if net_name in ['lstm', 'BiLSTM', 'GRU']:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            for i in range(int(IMAGE_NUM/BATCH_SIZE)):
                # forward
                inputs, labels = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], results[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                inputs = inputs.unsqueeze(dim=1)
                # inputs = inputs.unsqueeze(dim=2)
                optimizer.zero_grad()
                outputs = net(inputs)
                outputs = reshape_tensor(outputs, (5, 1))
                labels = labels.unsqueeze(dim=1)
                loss = criterion(outputs, labels)
                optimizer.step(loss)
            # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
            return
        criterion = nn.CrossEntropyLoss()
        # criterion.to('cuda')
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # for param in optimizer.param_groups:
        #     for key, value in param.items():
        #         if isinstance(value, torch.Tensor):
        #             param[key] = value.to('cuda')
        running_loss = 0.0

        for i in range(int(IMAGE_NUM/BATCH_SIZE)):
            # forward:
            inputs, labels = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], results[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            optimizer.zero_grad()
            outputs = net(inputs)

            # outputs reshape:
            if net_name in ['lenet', 'alexnet', 'vgg16', 'vgg19']:
                target_shape = (BATCH_SIZE, 10)
            else:
                target_shape = (BATCH_SIZE, 1000)
            outputs = reshape_tensor(outputs, target_shape)

            # loss calculation and backward:
            loss = criterion(outputs, labels)
            optimizer.step(loss)
            # running_loss += loss.item()
            # print(str(i) + ':  ' + str(loss.item()))
        # if os.environ['TRAIN_STOP_FLAG'] == '0':
        #     self.train_count += 1
        # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
        # torch.cpu.empty_cache()
        return


# class Tester:
#     def __init__(self):
#         MODEL_LIST = ['ResNet18', 'ResNet50', 'InceptionV3', 'xception', 'testnet',
#                       'alexnet', 'lenet', 'mobilenet', 'squeezenet', 'vgg16', 'vgg19',
#                       'densenet', 'LSTM', 'GRU', 'BiLSTM', 'googlenet']
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
#         self.dataloader_dict['LSTM'] = get_stock_price_test()
#         self.dataloader_dict['GRU'] = self.dataloader_dict['LSTM']
#         self.dataloader_dict['BiLSTM'] = self.dataloader_dict['LSTM']
#         self.dataloader_dict['googlenet'] = self.dataloader_dict['ResNet18']
#         return
#
#     def test(self, net_to_train: nn.Module, net_name: str) -> float:
#         if net_name in self.dataloader_dict.keys():
#             dataloader = self.dataloader_dict[net_name]  # now, data_loader is a tuple: (images, labels)
#         else:
#             dataloader = None
#         if dataloader is None:
#             # print(str(net_to_train).split('(', 1)[0] + ' no train')
#             return 0.0
#         net = net_to_train
#         images = dataloader[0]
#         results = dataloader[1]
#         if net_name in ['LSTM', 'BiLSTM', 'GRU']:
#             criterion = nn.MSELoss()
#             optimizer = optim.Adam(net.parameters(), lr=0.001)
#             correct = 0
#             total = 0
#             for i in range(int(IMAGE_NUM/BATCH_SIZE)):
#                 # forward
#                 inputs, labels = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], results[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
#                 inputs = inputs.unsqueeze(dim=1)
#                 inputs = inputs.unsqueeze(dim=2)
#                 outputs = net(inputs)
#                 outputs = reshape_tensor(outputs, (5, 1))
#                 labels = labels.unsqueeze(dim=1)
#                 _, predicted = jittor.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#             # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
#             return correct / total
#         criterion = nn.CrossEntropyLoss()
#         # criterion.to('cuda')
#         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#         correct = 0
#         total = 0
#         # for param in optimizer.param_groups:
#         #     for key, value in param.items():
#         #         if isinstance(value, torch.Tensor):
#         #             param[key] = value.to('cuda')
#         running_loss = 0.0
#
#         for i in range(int(IMAGE_NUM/BATCH_SIZE)):
#             # forward:
#             inputs, labels = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], results[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
#             outputs = net(inputs)
#
#             # outputs reshape:
#             if net_name in ['lenet', 'alexnet', 'vgg16', 'vgg19']:
#                 target_shape = (BATCH_SIZE, 10)
#             else:
#                 target_shape = (BATCH_SIZE, 1000)
#             outputs = reshape_tensor(outputs, target_shape)
#
#             # loss calculation and backward:
#             _, predicted = jittor.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # running_loss += loss.item()
#             # print(str(i) + ':  ' + str(loss.item()))
#         if os.environ['TRAIN_STOP_FLAG'] == '0':
#             self.train_count += 1
#         # print('Training Finished, ' + str(net).split('(', 1)[0] + ', count ' + str(self.train_count))
#         # torch.cpu.empty_cache()
#         return correct / total

jittor_trainer = Trainer()
# if __name__ == '__main__':
#     # test
#     t = Tester()
#
#
#     def traintrain(net_name: str):
#         module = import_module(net_name)
#         net = module.go()
#         t.test(net, net_name)
