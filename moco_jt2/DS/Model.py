import copy
import sys
import numpy
import random

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
                   f"import jittor.optim as optim\n" \
                   f"import numpy as np\n" \
                   f"import copy\n"\
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
        # ops_list = ['jittor.ops.arccos', 'jittor.ops.arccosh', 'jittor.ops.negative', 'jittor.ops.acosh',
        #             'jittor.ops.cosh',
        #             'jittor.ops.acos', 'jittor.ops.floor', 'jittor.ops.floor_int', 'jittor.ops.arcsin',
        #             'jittor.ops.arcsinh',
        #             'jittor.ops.asin', 'jittor.ops.asinh', 'jittor.ops.sigmoid', 'jittor.ops.cos',
        #             'jittor.ops.ceil',
        #             'jittor.ops.ceil_int', 'jittor.ops.sin', 'jittor.ops.sinh', 'jittor.ops.erf',
        #             'jittor.ops.erfinv',
        #             'jittor.ops.log', 'jittor.ops.atan', 'jittor.ops.atanh', 'jittor.ops.arctan',
        #             'jittor.ops.arctanh',
        #             'jittor.ops.abs', 'jittor.ops.sqrt', 'jittor.ops.tanh', 'jittor.ops.tan', 'jittor.ops.exp']
        ops_list = ['jittor.ops.arccos', 'jittor.ops.arccosh', 'jittor.ops.negative', 'jittor.ops.acosh',
                    'jittor.ops.cosh',
                    'jittor.ops.acos', 'jittor.ops.floor', 'jittor.ops.arcsin',
                    'jittor.ops.arcsinh',
                    'jittor.ops.asin', 'jittor.ops.asinh', 'jittor.ops.sigmoid', 'jittor.ops.cos',
                    'jittor.ops.ceil',
                    'jittor.ops.sin', 'jittor.ops.sinh', 'jittor.ops.erf',
                    'jittor.ops.erfinv',
                    'jittor.ops.log', 'jittor.ops.atan', 'jittor.ops.atanh', 'jittor.ops.arctan',
                    'jittor.ops.arctanh',
                    'jittor.ops.abs', 'jittor.ops.sqrt', 'jittor.ops.tanh', 'jittor.ops.tan', 'jittor.ops.exp']
        # var_list = ['jittor.Var.arccos()', 'jittor.Var.arccosh()', 'jittor.Var.negative()', 'jittor.Var.acosh()',
        #             'jittor.Var.cosh()', 'jittor.Var.acos()', 'jittor.Var.floor()', 'jittor.Var.floor_int()',
        #             'jittor.Var.arcsin()', 'jittor.Var.arcsinh()', 'jittor.Var.asin()', 'jittor.Var.asinh()',
        #             'jittor.Var.sigmoid()', 'jittor.Var.cos()', 'jittor.Var.ceil()', 'jittor.Var.ceil_int()',
        #             'jittor.Var.sin()', 'jittor.Var.sinh()', 'jittor.Var.erf()', 'jittor.Var.erfinv()',
        #             'jittor.Var.log()',
        #             'jittor.Var.atan()', 'jittor.Var.atanh()', 'jittor.Var.arctan()', 'jittor.Var.arctanh()',
        #             'jittor.Var.abs()', 'jittor.Var.sqrt()', 'jittor.Var.tanh()', 'jittor.Var.tan()',
        #             'jittor.Var.exp()']
        var_list = ['jittor.Var.arccos()', 'jittor.Var.arccosh()', 'jittor.Var.negative()', 'jittor.Var.acosh()',
                    'jittor.Var.cosh()', 'jittor.Var.acos()', 'jittor.Var.floor()',
                    'jittor.Var.arcsin()', 'jittor.Var.arcsinh()', 'jittor.Var.asin()', 'jittor.Var.asinh()',
                    'jittor.Var.sigmoid()', 'jittor.Var.cos()', 'jittor.Var.ceil()',
                    'jittor.Var.sin()', 'jittor.Var.sinh()', 'jittor.Var.erf()', 'jittor.Var.erfinv()',
                    'jittor.Var.log()',
                    'jittor.Var.atan()', 'jittor.Var.atanh()', 'jittor.Var.arctan()', 'jittor.Var.arctanh()',
                    'jittor.Var.abs()', 'jittor.Var.sqrt()', 'jittor.Var.tanh()', 'jittor.Var.tan()',
                    'jittor.Var.exp()']
        chosen_list = random.choice([ops_list, var_list])
        chosen_element = random.choice(chosen_list)
        if chosen_list == ops_list:
            process_input_data = [f"jt.{chosen_element.split('.')[-1]}(input_c)", f"jt.{chosen_element.split('.')[-1]}(input_g)"]
        else:
            process_input_data = [f"input_c.{chosen_element.split('.')[-1].split('(')[0]}()", f"input_g.{chosen_element.split('.')[-1].split('(')[0]}()"]

        code = f"def chebyshev_distance(A: np.ndarray, B: np.ndarray):\n" \
               f"    if A is None or B is None:\n" \
               f"        return 0.0\n" \
               f"    if A.shape != B.shape:\n" \
               f"        return 9999999\n" \
               f"    else:\n" \
               f"        return float(np.max(np.abs(A - B)))\n" \
               f"\n" \
               f"\n" \
               f"def train(x, x_t, y_t):\n" \
               f"    flag = True\n" \
               f"    jt.flags.use_cuda = 0\n" \
               f"    m_c = {self.modelName}()\n"\
               f"    opt_c = optim.SGD(m_c.parameters(), lr=0.01)\n"\
               f"\n"\
               f"    jt.flags.use_cuda = 1\n"\
               f"    m_g = copy.deepcopy(m_c)\n" \
               f"    opt_g = optim.SGD(m_g.parameters(), lr=0.01)\n" \
               f"\n" \
               f"    jt.flags.use_cuda = 0\n" \
               f"    input_c = jt.array(x_t).float32()\n"\
               f"    input_c = {process_input_data[0]}\n" \
               f"    target_c = jt.array(y_t)\n" \
               f"    output_c = m_c(input_c)\n" \
               f"    loss_c = nn.CrossEntropyLoss()(output_c, target_c)\n" \
               f"    opt_c.backward(loss_c)\n" \
               f"\n" \
               f"    jt.flags.use_cuda = 1\n" \
               f"    input_g = jt.array(x_t).float32()\n"\
               f"    input_g = {process_input_data[1]}\n" \
               f"    target_g = jt.array(y_t)\n" \
               f"    output_g = m_g(input_g)\n" \
               f"    loss_g = nn.CrossEntropyLoss()(output_g, target_g)\n"\
               f"    opt_g.backward(loss_g)\n"\
               f"\n" \
               f"    jt.flags.use_cuda = 0\n" \
               f"    if chebyshev_distance(output_c.detach().numpy(), output_g.detach().numpy()) > 0.1:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Output diff too big'\n" \
               f"    if abs(loss_c.item() - loss_g.item()) > 0.1:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Loss diff too big'\n" \
               f"    for (param_c, param_g) in zip(m_c.parameters(), m_g.parameters()):\n" \
               f"        weights_c = param_c\n" \
               f"        weights_g = param_g\n" \
               f"        distance = chebyshev_distance(weights_c, weights_g)\n" \
               f"        if distance > 0.1:\n" \
               f"            flag = False\n" \
               f"            break\n" \
               f"    if not flag:\n" \
               f"        return flag, 'Grad diff too big'\n" \
               f"\n" \
               f"    return flag, ''\n"
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
