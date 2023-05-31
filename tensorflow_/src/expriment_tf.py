from pathlib import Path

from config.paths import tf_res_file
from tensorflow_.src.mutate_tf import MoCoTF
from utils.Experiments import Experiments


class ExperimentsTF(Experiments):

    def __init__(self):
        super(ExperimentsTF, self).__init__()
        self.model = self.simple_model + self.complex_model

    def departOne(self, model):
        if (tf_res_file / model).exists():
            print("文件存在")
            return
        mocoTf = MoCoTF(model)
        mocoTf.depart()
        print(Path(model).name + "\033[92m分解完成\033[0m")

    def departAll(self):
        for model in self.simple_model:
            if (tf_res_file / model).exists():
                print("文件存在")
                continue
            mocoTf = MoCoTF(model)
            mocoTf.depart()
            print(Path(model).name + "\033[92m分解完成\033[0m")

        for model in self.complex_model:
            if (tf_res_file / model).exists():
                print("文件存在")
                continue
            mocoTf = MoCoTF(model)
            mocoTf.depart()
            print(Path(model).name + "\033[92m分解完成\033[0m")

    def mutateOne(self, model):
        mocoTf = MoCoTF(model)
        mocoTf.generate_model()
        print(mocoTf.error_list)

    def mutateAll(self): pass

    def trainOne(self):
        # model = random.choice(self.model)
        model = self.model[0]
        self.departOne(model)
        self.mutateOne(model)

    def trainAll(self): pass


if __name__ == "__main__":
    exp = ExperimentsTF()
    exp.departOne("inceptionv3")
    exp.mutateOne("inceptionv3")

