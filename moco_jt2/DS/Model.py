import copy
import sys

import numpy

from moco_jt2.DS.Block import Block
from moco_jt2.Tools.ConstraintChecker import CheckBlock


class Model:
    def __init__(self, graph, modelInputs, modelOutputs):
        self.modelName = ""
        self.graph: list[Block] = graph
        self.modelInputs: list[str] = modelInputs
        self.modelOutputs: list[str] = modelOutputs
        self.oracle = None  # Type: Oracle
        return

    def SetOracle(self, oracle):
        self.oracle = oracle

    def AssembleModel(self, extraModelInputs=""):
        decl_bound = "\n".join([b.GenerateDeclarationStatement() for b in self.graph])
        forward_bound = "\n".join([b.GenerateForwardStatement() for b in self.graph])
        code = f"class {self.modelName}(nn.Module):\n" \
               f"    def __init__(self{extraModelInputs}):\n" \
               f"        super().__init__()\n" \
               f"{decl_bound}\n" \
               f"    \n" \
               f"    def execute(self, {','.join(self.modelInputs)}):\n" \
               f"{forward_bound}\n" \
               f"        return {','.join(self.modelOutputs)}\n"
        return code

    def AssembleFile(self, outputPath, fileName, hasGoCode=True, hasTrainCode=False, useGPU=False):  # fileName here has no .py
        childDone = []
        mainModelCode = self.AssembleModel()
        childModelCode = "\n"
        for block in self.graph:
            if block.isChildModel:
                if block.apiName not in childDone:
                    childDone.append(block.apiName)
                    # extraModelInputs = ", " + ", ".join(list(block.params.keys()))
                    extraModelInputs = ""
                    childModelCode += block.childModel.AssembleModel(extraModelInputs)
                    childModelCode += "\n\n"
        fileCode = f"import jittor\n" \
                   f"import jittor as jt\n"\
                   f"import jittor.nn as nn\n"\
                   f"import jittor.optim as optim\n"\
                   f"import copy\n" \
                   f"\n" \
                   f"\n" \
                   f"{mainModelCode}\n" \
                   f"{childModelCode}\n" \
                   f"\n{self.GenerateGoCode(useGPU) if hasGoCode else ''}\n" \
                   f"\n{self.GenerateTrainCode() if hasTrainCode else ''}\n"
        f = open(f"{outputPath}/{fileName}.py", "w", encoding="utf-8")
        f.write(fileCode)
        f.close()
        return f"{outputPath}/{fileName}.py"

    def GenerateGoCode(self, useGPU=False):
        gpu_str = "jt.flags.use_cuda = 1" if useGPU else ""
        code = f"def go():\n" \
               f"    {gpu_str}\n" \
               f"    x = jittor.randn({self.GetInputShapeStr()})\n" \
               f"    m = {self.modelName}()\n" \
               f"    y = m(x)\n" \
               f"    return list(y.shape)\n\n\n"
        return code

    def GenerateTrainCode(self):
        code = f"def train(x, x_t, y_t):\n" \
               f"    jt.flags.use_cuda = 0\n" \
               f"    m_c = {self.modelName}()\n"\
               f"    opt_c = optim.SGD(m_c.parameters(), lr=0.01)\n"\
               f"\n"\
               f"    jt.flags.use_cuda = 1\n"\
               f"    m_g = copy.deepcopy(m_c)\n" \
               f"    opt_g = optim.SGD(m_g.parameters(), lr=0.01)\n" \
               f"\n" \
               f"    jt.flags.use_cuda = 0\n" \
               f"    input_c = jt.array(x_t).float32()\n" \
               f"    target_c = jt.array(y_t)\n" \
               f"    output_c = m_c(input_c)\n" \
               f"    loss_c = nn.CrossEntropyLoss()(output_c, target_c)\n" \
               f"    opt_c.backward(loss_c)\n" \
               f"    opt_c.step()\n" \
               f"    gradients_c = [p.opt_grad(opt_c).numpy() for p in m_c.parameters()]\n" \
               f"\n" \
               f"    jt.flags.use_cuda = 1\n" \
               f"    input_g = jt.array(x_t).float32()\n" \
               f"    target_g = jt.array(y_t)\n" \
               f"    output_g = m_g(input_g)\n" \
               f"    loss_g = nn.CrossEntropyLoss()(output_g, target_g)\n"\
               f"    opt_g.backward(loss_g)\n"\
               f"    opt_g.step()\n" \
               f"    gradients_g = [p.opt_grad(opt_g).numpy() for p in m_g.parameters()]\n"\
               f"\n"\
               f"    jt.flags.use_cuda = 0\n"\
               f"    x_c = jt.array(x).float32()\n" \
               f"    y_c = m_c(x_c)\n"\
               f"\n"\
               f"    jt.flags.use_cuda = 1\n"\
               f"    x_g = jt.array(x).float32()\n" \
               f"    y_g = m_g(x_g)\n"\
               f"\n" \
               f"    output_comparison = y_c.detach().numpy(), y_g.detach().numpy()\n" \
               f"    loss_comparison = loss_c.item(), loss_g.item()\n" \
               f"    gradients_comparison = gradients_c, gradients_g\n" \
               f"\n" \
               f"    return output_comparison, loss_comparison, gradients_comparison\n"
        return code

    def GenerateFuncCode(self):
        # TODO: 单算子函数，在class外，和go一起
        pass

    def GetInputShapeStr(self):
        if self.modelName in ["lenet"]:
            return "[1, 1, 28, 28]"
        elif self.modelName in ["LSTM"]:
            return "[1, 5, 1]"
        elif self.modelName in ["pointnet"]:
            return "[2, 3, 2048]"
        else:
            return "[1, 3, 224, 224]"

    def ModelPreHandle(self):
        self.CalculateShape()
        for block in self.graph:
            block.ChildModelAssign()
            block.ChildModelInitializeShape()
        return

    def CalculateShape(self):
        inceptions = [192, 256, 480, 512, 512, 512, 528, 832, 832, 1024]
        i, o = 0, 1
        pre = None
        for block in self.graph:
            if not block.isChildModel:
                if "Linear" in block.apiName:
                    block.InitializeShape(None)
                    block.SetShape(d=2)
                else:
                    block.InitializeShape(pre)
            else:
                block.SetShape(inceptions[i], inceptions[o], pre.dim)
                i += 1
                o += 1
            pre = copy.deepcopy(block)

    def CalculateShapeChildModel(self, firstIn, finalOut):
        pre = None
        for block in self.graph:
            if not block.isChildModel:
                if "Linear" in block.apiName:
                    block.InitializeShape(None)
                    block.SetShape(d=2)
                else:
                    block.InitializeShape(pre)
            else:
                pass
            pre = copy.deepcopy(block)
        if self.graph[0].inChannels == 0:
            self.graph[0].SetShape(firstIn, firstIn, 2 if "1d" not in self.graph[0].apiName.lower() else 1)
            for i, block in enumerate(self.graph):
                if i == 0:
                    continue
                if not block.isChildModel:
                    if "Linear" in block.apiName:
                        block.InitializeShape(None)
                        block.SetShape(d=2)
                    else:
                        block.InitializeShape(pre)
                else:
                    pass
                pre = copy.deepcopy(block)
