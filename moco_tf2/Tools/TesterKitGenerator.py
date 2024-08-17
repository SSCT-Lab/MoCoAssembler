import numpy as np
import random


dataset_folder = "/Users/wuduo/Documents/BioWork/MoCo/MOCO-7/datasets"
DATASET_PATHS = {
    "LeNet": f"{dataset_folder}/MNIST.npz",
    "Common224": f"{dataset_folder}/imagenet.npz",
    "PointNet": f"{dataset_folder}/modelnet10.npz"
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
            self.dataset_x = self.dataset_x.reshape(60000, 28, 28, 1) / 255.0
            self.dataset_x = self.dataset_x.astype(np.float32)
            # 归一化
            self.length = 200
            return
        elif seed_name in ["alexnet", "mobilenet", "squeezenet", "vgg19", "GoogleNet"]:
            dataset_path = DATASET_PATHS["Common224"]
            data = np.load(dataset_path)
            self.dataset_x, self.dataset_y = data["x_test"], data["y_test"]
            self.dataset_x = self.dataset_x / 255.0
            self.dataset_y = self.dataset_y.astype(np.uint8)
            self.length = 1500
        elif seed_name in ["pointnet", "lstm"]:
            dataset_path = DATASET_PATHS["PointNet"]
            data = np.load(dataset_path)
            self.dataset_x, self.dataset_y = data["x_train"], data["y_train"]
            self.dataset_y = self.dataset_y.astype(np.uint8)
            self.length = 3991

    def generate_kit(self, batch_size=1):
        start = random.randint(0, self.length-batch_size)
        return self.dataset_x[start: start+batch_size], self.dataset_y[start: start+batch_size]


test_kit_generator = None
