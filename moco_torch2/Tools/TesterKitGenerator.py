import numpy as np
import random


datasetFolder = "./Data/Datasets"
DATASET_PATHS = {
    "LeNet": f"{datasetFolder}/MNIST.npz",
    "Common224": f"{datasetFolder}/imagenet.npz",
    "PointNet": f"{datasetFolder}/modelnet10.npz"
}


class TestKitGenerator:
    def __init__(self, seed_name):
        if seed_name == "":
            return
        self.seed_name = seed_name
        self.dataset_x = None
        self.dataset_y = None
        self.length = 0

        if seed_name == "lenet":
            dataset_path = DATASET_PATHS["LeNet"]
            data = np.load(dataset_path)
            self.dataset_x, self.dataset_y = data["x_train"], data["y_train"]
            self.dataset_x = self.dataset_x.reshape(60000, 1, 28, 28)
            self.dataset_x = self.dataset_x.astype(np.float32)
            # 归一化
            self.dataset_x = self.dataset_x / 255.0
            self.length = 200
            return
        elif seed_name in ["alexnet", "mobilenet", "squeezenet", "vgg19", "googlenet", "resnet18"]:
            dataset_path = DATASET_PATHS["Common224"]
            data = np.load(dataset_path)
            self.dataset_x, self.dataset_y = data["x_test"], data["y_test"]
            self.dataset_x = np.transpose(self.dataset_x, (0, 3, 1, 2))
            self.dataset_x = self.dataset_x / 255.0
            self.dataset_y = self.dataset_y.astype(np.uint8)
            self.length = 1500
        elif seed_name in ["pointnet", "LSTM"]:
            dataset_path = DATASET_PATHS["PointNet"]
            data = np.load(dataset_path)
            self.dataset_x, self.dataset_y = data["x_train"], data["y_train"]
            self.dataset_x = np.transpose(self.dataset_x, (0, 2, 1))
            self.dataset_y = self.dataset_y.astype(np.uint8)
            self.length = 3991

    def generate_kit(self, batch_size=1):
        start = random.randint(0, self.length-batch_size)
        return self.dataset_x[start: start+batch_size], self.dataset_y[start: start+batch_size], start


test_kit_generator = None
