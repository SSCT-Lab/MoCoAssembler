import copy
import numpy

from DS.Block import Block
from Tools.ConstraintChecker import CheckBlock


class Model:
    def __init__(self, graph, modelInputs, modelOutputs):
        self.modelName = ""
        self.graph = graph
        self.modelInputs = modelInputs
        self.modelOutputs = modelOutputs
        self.oracle = None  # Type: Oracle
        return

    def SetOracle(self, oracle):
        self.oracle = oracle

    def AssembleModel(self, extraModelInputs=""):
        decl_bound = "\n".join([b.GenerateDeclarationStatement() for b in self.graph])
        forward_bound = "\n".join([b.GenerateForwardStatement() for b in self.graph])
        code = f"class {self.modelName}(nn.Block):\n" \
               f"    def __init__(self, **kwargs):\n" \
               f"        super().__init__(**kwargs)\n" \
               f"{decl_bound}\n" \
               f"    \n" \
               f"    def forward(self, {','.join(self.modelInputs)}):\n" \
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
        fileCode = f"import mxnet\n" \
                   f"import mxnet.gluon.nn as nn\n" \
                   f"import mxnet as mx\n" \
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
        code = f"def go():\n" \
               f"    x = mx.nd.random.uniform(shape=({self.GetInputShapeStr()}), ctx=mx.cpu())\n" \
               f"    m = {self.modelName}()\n" \
               f"    m.initialize(mx.init.Xavier(), ctx=mx.cpu())\n" \
               f"    y = m(x)\n" \
               f"    return m\n\n\n"
        return code

    def GenerateTrainCode(self):
        code = f"def train(x, x_t, y_t):\n" \
               f"    m_c = {self.modelName}().to('cpu')\n" \
               f"    m_g = copy.deepcopy(m_c)\n" \
               f"    m_g.to('cuda')\n" \
               f"    input_c = mxnet.from_numpy(x_t).to(mxnet.float32).to('cpu')\n" \
               f"    input_g = mxnet.from_numpy(x_t).to(mxnet.float32).to('cuda')\n" \
               f"    target_c = mxnet.from_numpy(y_t).to('cpu')\n" \
               f"    target_g = mxnet.from_numpy(y_t).to('cuda')\n" \
               f"    output_c = m_c(input_c)\n" \
               f"    loss_c = nn.CrossEntropyLoss()(output_c, target_c)\n" \
               f"    loss_c.backward()\n" \
               f"    output_g = m_g(input_g)\n" \
               f"    loss_g = nn.CrossEntropyLoss()(output_g, target_g)\n" \
               f"    loss_g.backward()\n" \
               f"    x_c = mxnet.from_numpy(x).to(mxnet.float32).to('cpu')\n" \
               f"    y_c = m_c(x_c)\n" \
               f"    x_g = mxnet.from_numpy(x).to(mxnet.float32).to('cuda')\n" \
               f"    y_g = m_g(x_g)\n" \
               f"\n" \
               f"    return y_c.detach().to('cpu').numpy(), y_g.detach().to('cpu').numpy()\n"
        return code

    def GenerateFuncCode(self):
        # TODO: 单算子函数，在class外，和go一起
        pass

    def GetInputShapeStr(self):
        if self.modelName in ["LeNet"]:
            return "1, 1, 32, 32"
        elif self.modelName in ["LSTM"]:
            return "[1, 5, 1]"
        elif self.modelName in ["PointNet"]:
            return "3, 3, 100"
        elif self.modelName in ["SqueezeNet"]:
            return "1, 3, 224, 224"
        else:
            return "1, 3, 227, 227"

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
