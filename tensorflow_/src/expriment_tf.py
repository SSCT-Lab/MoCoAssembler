import random
from pathlib import Path

from config.paths import RES_PATH
from tensorflow_.src.mutate_tf import MoCoTF
from utils.Experiments import Experiments


class ExperimentsTF(Experiments):

    def __init__(self):
        super(ExperimentsTF, self).__init__()
        self.model = self.simple_model + self.complex_model

    def departOne(self, model):
        if (RES_PATH / model).exists():
            print(model + "文件存在")
            return
        mocoTf = MoCoTF(model)
        mocoTf.depart()
        print(model + "\033[92m分解完成\033[0m")

    def departAll(self):
        for model in self.simple_model:
            if (RES_PATH / model).exists():
                print(model + "文件存在")
                continue
            mocoTf = MoCoTF(model)
            mocoTf.depart()
            print(model + "\033[92m分解完成\033[0m")

        for model in self.complex_model:
            if (RES_PATH / model).exists():
                print("文件存在")
                continue
            mocoTf = MoCoTF(model)
            mocoTf.depart()
            print(model + "\033[92m分解完成\033[0m")

    def mutateOne(self, model):
        mocoTf = MoCoTF(model)
        mocoTf.generate_model()
        print(mocoTf.error_list)

    def mutateAll(self): pass

    def trainOne(self, model):
        self.departOne(model)
        self.mutateOne(model)

    def trainAll(self): pass


if __name__ == "__main__":
    exp = ExperimentsTF()
    # simple_model_list = exp.simple_model
    # for model in simple_model_list:
    #     print(model + "\033[94mSTART\033[0m")
    #     exp.trainOne(model)
    #     print(model + "\033[94mDONE\033[0m")

    exp.trainOne("lenet")

