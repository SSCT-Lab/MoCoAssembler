import copy

from DS.Block import Block, PARAM_NEED_OP


def GetLSTM():
    lstm1 = Block(
        "torch.nn.LSTMCell",
        {"input_size": 2048, "hidden_size": 200},
        "lstm1",
        ["x"],
        ["x"]
    )
    lstm1.inChannels = 2048
    lstm1.outChannels = 200
    lstm1.dim = 1

    relu1 = Block(
        "torch.nn.ReLU",
        {},
        "relu1",
        ["x"],
        ["x"]
    )
    relu1.inChannels = 200
    relu1.outChannels = 200
    relu1.dim = 1

    lstm2 = copy.deepcopy(lstm1)
    lstm2.params["input_size"] = 200
    lstm2.inChannels = 200
    lstm2.nodeName = "lstm2"
    lstm3 = copy.deepcopy(lstm2)
    lstm3.nodeName = "lstm3"
    lstm4 = copy.deepcopy(lstm3)
    lstm4.nodeName = "lstm4"

    relu2 = copy.deepcopy(relu1)
    relu2.nodeName = "relu2"
    relu3 = copy.deepcopy(relu2)
    relu3.nodeName = "relu3"
    relu4 = copy.deepcopy(relu3)
    relu4.nodeName = "relu4"

    linear = Block(
        "torch.nn.Linear",
        {"in_features": 200, "out_features": 10},
        "fc",
        ["x"],
        ["x"]
    )
    linear.inChannels = 200
    linear.outChannels = 10
    linear.dim = 1

    graph = [lstm1, relu1, lstm2, relu2, lstm3, relu3, lstm4, relu4, linear]
    lstm = RNNModel(graph, ['x'], ['x'])
    lstm.modelName = "LSTM"
    return lstm


class RNNModel:
    def __init__(self, graph, modelInputs, modelOutputs):
        self.modelName = ""
        self.graph: list[Block] = graph
        self.modelInputs: list[str] = modelInputs
        self.modelOutputs: list[str] = modelOutputs
        self.oracle = None  # Type: Oracle
        return

    def SetOracle(self, oracle):
        self.oracle = oracle

    def GenerateForwardBound(self):
        res = ""
        i = 1
        inFor = True
        # core = min(1 + int(len(self.graph) / 2 - 0.2), 4)
        for block in self.graph:
            if block.apiName in ["torch.nn.LSTMCell", "torch.nn.GRUCell", "torch.nn.RNNCell"]:
                if block.apiName == "torch.nn.LSTMCell":
                    res += f"            hn{i}, cn{i} = self.{block.nodeName}(hn{i-1}, (hn{i}, cn{i}))\n"
                else:
                    res += f"            hn{i} = self.{block.nodeName}(hn{i-1}, hn{i})\n"
                i = i + 1
            elif (block.apiName == "torch.nn.Linear" or block.apiName == "torch.nn.Flatten") and inFor:
                res += f"        x = self.{block.nodeName}(hn{i-1})\n"
                inFor = False
            else:
                if inFor:
                    if block.blockType == PARAM_NEED_OP:
                        res += f"            hn{i-1} = self.{block.nodeName}(hn{i-1}, {block.GenerateParamString()})\n"
                    else:
                        res += f"            hn{i-1} = self.{block.nodeName}(hn{i-1})\n"
                else:
                    res += block.GenerateForwardStatement()
                    res += "\n"
        if len(self.graph) == 0 or self.graph[-1].apiName == "torch.nn.Linear":
            pass
        else:
            res += f"        x = hn{i-1}\n"
        return res

    def AssembleModel(self, extraModelInputs=""):
        decl_bound = "\n".join([b.GenerateDeclarationStatement() for b in self.graph])
        forward_bound = self.GenerateForwardBound()

        code = f"class {self.modelName}(nn.Module):\n" \
               f"    def __init__(self{extraModelInputs}):\n" \
               f"        super().__init__()\n" \
               f"        self.hidden_size = 200\n" \
               f"        self.h0 = torch.zeros((1, self.hidden_size))\n" \
               f"        self.c0 = torch.zeros((1, self.hidden_size))\n" \
               f"{decl_bound}\n" \
               f"    \n" \
               f"    def forward(self, x):\n" \
               f"        seq_length = x.size(1)\n" \
               f"        hn1, cn1 = self.h0, self.c0\n" \
               f"        hn2, cn2 = self.h0, self.c0\n" \
               f"        hn3, cn3 = self.h0, self.c0\n" \
               f"        hn4, cn4 = self.h0, self.c0\n" \
               f"        for t in range(seq_length):\n" \
               f"            hn0 = x[:, t, :]\n" \
               f"{forward_bound}\n" \
               f"        return x\n"
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
        fileCode = f"import torch\n" \
                   f"import torch.nn as nn\n" \
                   f"import copy\n" \
                   f"import numpy as np\n" \
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
        gpu_str = ".to('cuda')" if useGPU else ".to('cpu')"
        code = f"def go():\n" \
               f"    x = torch.randn({self.GetInputShapeStr()}){gpu_str}\n" \
               f"    m = {self.modelName}(){gpu_str}\n" \
               f"    m.h0 = m.h0{gpu_str}\n" \
               f"    m.c0 = m.c0{gpu_str}\n" \
               f"    y = m(x)\n" \
               f"    return list(y.shape)\n\n\n"
        return code

    def GenerateTrainCode(self):
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
               f"    m_c = {self.modelName}().to('cpu')\n" \
               f"    m_g = copy.deepcopy(m_c)\n" \
               f"    m_g.to('cuda')\n" \
               f"    m_g.h0 = m_g.h0.to('cuda')\n" \
               f"    m_g.c0 = m_g.c0.to('cuda')\n" \
               f"    input_c = torch.from_numpy(x_t).to(torch.float32).to('cpu')\n" \
               f"    input_g = torch.from_numpy(x_t).to(torch.float32).to('cuda')\n" \
               f"    target_c = torch.from_numpy(y_t).to('cpu')\n" \
               f"    target_g = torch.from_numpy(y_t).to('cuda')\n" \
               f"    output_c = m_c(input_c)\n" \
               f"    loss_c = nn.CrossEntropyLoss()(output_c, target_c)\n" \
               f"    loss_c.backward()\n" \
               f"    output_g = m_g(input_g)\n" \
               f"    loss_g = nn.CrossEntropyLoss()(output_g, target_g)\n" \
               f"    loss_g.backward()\n" \
               f"\n" \
               f"    if chebyshev_distance(output_c.detach().to('cpu').numpy(), output_g.detach().to('cpu').numpy()) > 0.1:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Output diff too big'\n" \
               f"    if abs(loss_c.item() - loss_g.item()) > 0.1:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Loss diff too big'\n" \
               f"    for (param_c, param_g) in zip(m_c.parameters(), m_g.parameters()):\n" \
               f"        weights_c = param_c.cpu().detach().numpy()\n" \
               f"        weights_g = param_g.cpu().detach().numpy()\n" \
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
            return "[1, 3, 2048]"
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
