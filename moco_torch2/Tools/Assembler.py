import copy
import pickle
import os
import time
import json
import importlib.util
import sys
import traceback
import numpy as np
import heapq

from alive_progress import alive_bar
from collections import defaultdict
from DS.Model import Model
from DS.Block import Block
from Tools.Parser import GetSeed
from Tools.TesterKitGenerator import TestKitGenerator
from Tools.Mutator import Mutator

threshold = 0.1
mutator = Mutator()


class Filter:
    def __init__(self):
        self.info_lis = []
        self.tkg = None
        self.addkey("must be divisible by groups")
        # self.addkey("pad should be at most half of kernel size")
        self.addkey("missing 1 required positional argument")
        # self.addkey("Kernel size can't be greater than actual input size")
        self.addkey("out_channels must be divisible by groups")
        self.addkey("but got Tensor of dimension")
        self.addkey("mat1 and mat2 shapes cannot be multiplied")
        # self.addkey("Bilinear.forward() missing 1 required positional argument")
        # self.addkey("No module named")
        self.addkey("but got input of size")
        # self.addkey("syntax error")
        self.addkey("Output size is too small")
        self.addkey("expected to be in range of")
        self.addkey("missing 2 required positional arguments")
        self.addkey("channels instead")
        self.addkey("float() argument must be a string or a real number, not ")
        self.addkey("flatten() has invalid args: start_dim cannot come after end_dim")
        self.addkey("Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now")
        self.addkey("'tuple' object has no attribute")
        self.addkey("Expected 2D or 3D (batch mode) tensor for input")
        self.addkey("pool2d(): Expected")
        self.addkey("tensor expected for input")
        self.addkey("while checking arguments for")
        self.addkey("expects input with > 2 dims")
        self.addkey("must be tuple of ints, but found")
        self.addkey("Sizes of tensors must match")
        self.addkey("Tensors must have same number of dimensions")
        self.addkey("must be Tensor, not tuple")
        self.addkey("input has inconsistent input_size")
        self.addkey("input.size(-1) must be equal to input_size")
        self.addkey("It is expected dilation equals to 2")
        self.addkey("Input dimension should be at least 3")
        self.addkey("running_mean should contain")
        self.addkey("weight should contain")
        self.addkey("Expected weight to be")
        self.addkey("Padding length must be divisible by 2")
        self.addkey("It is expected stride equals to 2")
        self.addkey("expected 4D input")
        self.addkey("Expected more than 1 spatial element when training")
        self.addkey("Expected size of input")
        self.addkey("The size of tensor a")

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True

    def addkey(self, string: str) -> None:
        self.info_lis.append(string)


tailTemplate = GetSeed("tail").graph
flatten, out_10, out_1000 = tailTemplate[1], tailTemplate[2], tailTemplate[3]


filtor = Filter()
tkg = None


def b():
    print("=======================")


inputShapeTable = {
    "lenet": [1, 1, 28, 28],
    "mobilenet": [1, 3, 224, 224],
    "alexnet": [1, 3, 224, 224],
    "googlenet": [1, 3, 224, 224],
    "vgg19": [1, 3, 224, 224],
    "squeezenet": [1, 3, 224, 224],
    "pointnet": [2, 3, 2048]
}


def saveModelInFolder(model: Model, folder: str):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'model.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def loadModelInFolder(folder: str):
    file_path = os.path.join(folder, 'model.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model file found in {folder}")
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 100
    else:
        return float(np.max(np.abs(A - B)))


class TreeNode:
    def __init__(self):
        self.seedName = ""
        self.generation = 0
        self.index = 0
        self.father = None
        self.children = []
        self.basePath = ""
        self.casePath = ""
        self.outputShape = [0, 0, 0, 0]
        self.weight = 9999
        self.mutateInfo = ""

    def no(self):
        return self.generation, self.index

    def getFather(self):
        return self.father

    def setFather(self, father):
        self.father = father

    def getChildren(self):
        return self.children

    def newNode(self, basePath, generation, index, seedName):
        self.seedName = seedName
        self.basePath = basePath
        self.generation = generation
        self.index = index
        self.casePath = f"{basePath}/{generation}/{index}"
        os.makedirs(self.casePath, exist_ok=True)
        return

    def saveCase(self, model):
        if self.casePath == "":
            return
        else:
            saveModelInFolder(model, self.casePath)
            return

    def getModel(self):
        if self.casePath == "":
            return
        else:
            return loadModelInFolder(self.casePath)

    def assembleGoFile(self):
        filePath = self.casePath
        fileName = f"{self.seedName}_{self.generation}_{self.index}_go"
        model: Model = self.getModel()
        model.AssembleFile(
            outputPath=filePath,
            fileName=fileName,
            hasGoCode=True,
            hasTrainCode=False,
            useGPU=True
        )

    def assembleTrainFile(self):
        filePath = self.casePath
        fileName = f"{self.seedName}_{self.generation}_{self.index}_train"
        model: Model = self.getModel()
        if len(self.outputShape) >= 2:
            inChannels = 1
            for i in range(1, len(self.outputShape)):
                inChannels *= self.outputShape[i]
        elif len(self.outputShape) == 1:
            inChannels = self.outputShape[0]
        else:
            inChannels = 1
        tailFlatten: Block = copy.deepcopy(flatten)
        tailFlatten.nodeName = "tail_flatten"
        tailFlatten.inputSymbols = model.modelOutputs
        tailFc: Block = copy.deepcopy(out_10 if self.seedName in ["lenet", "pointnet"] else out_1000)
        tailFc.SetShape(inC=inChannels)
        tailFc.FixShape()
        tailFc.nodeName = "tail_fc"
        model.graph.append(tailFlatten)
        model.graph.append(tailFc)
        model.modelOutputs = tailFc.outputSymbols
        model.AssembleFile(
            outputPath=filePath,
            fileName=fileName,
            hasGoCode=True,
            hasTrainCode=True,
            useGPU=True
        )

    def run(self):
        self.assembleGoFile()
        filePath = f"{self.casePath}/{self.seedName}_{self.generation}_{self.index}_go.py"
        sys.path.append(self.casePath)
        start = time.time()
        try:
            module_name = f"{self.seedName}_{self.generation}_{self.index}_go"
            spec = importlib.util.spec_from_file_location(module_name, filePath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.outputShape = module.go()
            error_message = ""
            end = time.time()
        except Exception as e:
            end = start - 1
            error_message = traceback.format_exc()
        sys.path.remove(self.casePath)

        runTime = end - start
        return runTime, error_message

    def train(self):
        self.assembleTrainFile()
        x, _ = filtor.tkg.generate_kit()
        xt, yt = filtor.tkg.generate_kit()
        filePath = f"{self.casePath}/{self.seedName}_{self.generation}_{self.index}_train.py"
        sys.path.append(self.casePath)

        start = time.time()
        try:
            module_name = f"{self.seedName}_{self.generation}_{self.index}_go"
            spec = importlib.util.spec_from_file_location(module_name, filePath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            y1, y2 = module.train(x, xt, yt)
            error_message = ""
            end = time.time()
        except Exception as e:
            y1, y2 = None, None
            end = start - 1
            error_message = traceback.format_exc()

        sys.path.remove(self.casePath)
        trainTime = end - start

        if trainTime > 0 and error_message == "":
            cd = chebyshev_distance(y1, y2)
            if cd > threshold:
                trainTime = -1.0
                error_message = f"Diff too big: {str(cd)}"

        return trainTime, error_message

    def preCheck(self, block):
        code = f"import torch\n" \
               f"import torch.nn as nn\n" \
               f"\n" \
               f"\n" \
               f"def pre_check():\n" \
               f"    x = torch.randn({str(self.outputShape)})\n" \
               f"{block.GeneratePreCheckStatement()}\n"
        sys.path.append(f"{self.basePath}/../..")
        f = open(f"{self.basePath}/../../pre_check.py", "w", encoding="utf-8")
        f.write(code)
        f.close()
        try:
            module_name = "pre_check"
            spec = importlib.util.spec_from_file_location(module_name, f"{self.basePath}/../../pre_check.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            module.pre_check()
            error_message = ""
        except Exception as e:
            error_message = str(e)
        sys.path.remove(f"{self.basePath}/../..")
        if error_message == "":
            return True
        else:
            return filtor.judge(error_message)

    def spawn(self, n, block, startCount, b):
        # spawn n children based on block
        res = []
        count = startCount - 1
        fatherShape = self.outputShape
        if len(fatherShape) >= 2:
            inChannels = fatherShape[1]
        elif len(fatherShape) == 1:
            inChannels = fatherShape[0]
        else:
            inChannels = -1
        for i in range(n):
            count += 1
            child = TreeNode()
            child.newNode(self.basePath, self.generation + 1, count, self.seedName)
            child.father = self
            self.children.append(child)

            b()

            mutatedBlock, mutateInfo = mutator.Mutate(block)
            if inChannels != -1:
                mutatedBlock.SetShape(inC=inChannels)
                mutatedBlock.FixShape()
            retryCount = 0
            preCheckPassed = False
            while retryCount < 9:
                retryCount += 1
                if self.preCheck(mutatedBlock):
                    preCheckPassed = True
                    break
                else:
                    mutatedBlock, mutateInfo = mutator.Mutate(block)
                    if inChannels != -1:
                        mutatedBlock.SetShape(inC=inChannels)
                        mutatedBlock.FixShape()
            if not preCheckPassed:
                count += 1
                continue

            brandNewModel = self.getModel()
            brandNewModel.graph.append(mutatedBlock)
            brandNewModel.modelOutputs = mutatedBlock.outputSymbols
            child.saveCase(brandNewModel)
            child.mutateInfo = mutateInfo

            runTime, runErrorInfo = child.run()
            runResult = (runTime >= 0.0)
            if runResult:
                trainTime, trainErrorInfo = child.train()
                trainResult = (trainTime >= 0.0)
            else:
                trainTime, trainErrorInfo, trainResult = -1.0, "", False
            result = {
                "go result": runResult,
                "go time": runTime,
                "go error info": runErrorInfo,
                "train result": trainResult,
                "train time": trainTime,
                "train error info": trainErrorInfo,
                "father": str(self.no()),
                "mutate info": mutateInfo
            }
            f = open(f"{child.casePath}/result.json", "w", encoding="utf-8")
            json.dump(result, f, indent=2)
            f.close()

            if runResult and trainResult:
                child.weight = runTime + trainTime
                res.append(child)

        return res  # Here, model of these TreeNodes has been saved, and it just return fine nodes.


def cut(nodes, maxNode):
    if len(nodes) <= maxNode:
        return nodes

    # 将节点按等价类分类
    classes = defaultdict(list)
    for node in nodes:
        classes[node.mutateInfo].append(node)

    # 使用优先队列（最小堆）选择权重最大的节点进行削减
    # Python 的 heapq 是最小堆，所以用权重的负值来模拟最大堆
    max_heap = []
    for key, nodes in classes.items():
        for node in nodes:
            heapq.heappush(max_heap, (-node.weight, node, key))

    # 需要删除的节点数量
    num_to_remove = len(nodes) - maxNode

    # 保证不将任何类减少到零
    final_nodes = {key: nodes for key, nodes in classes.items()}
    while num_to_remove > 0:
        # 拿出一个最重的节点
        weight, node, mutate_info = heapq.heappop(max_heap)
        weight = -weight  # 转回正值

        # 确保不会完全删除某个等价类
        if len(final_nodes[mutate_info]) > 1:
            final_nodes[mutate_info].remove(node)
            num_to_remove -= 1

    # 重新构造结果列表
    result = []
    for nodes in final_nodes.values():
        result.extend(nodes)

    return result


class Assembler:
    def __init__(self, seed, n=2, maxEachLayer=500, experimentName=None, outputPath="./Output"):
        if experimentName is None:
            self.experimentName = seed + str(time.time())
        else:
            self.experimentName = experimentName
        self.baseOutputPath = f"{outputPath}/{experimentName}/tree"
        self.baseReportPath = f"{outputPath}/{experimentName}/report"
        os.makedirs(outputPath, exist_ok=True)
        os.makedirs(self.baseOutputPath, exist_ok=True)
        os.makedirs(self.baseReportPath, exist_ok=True)

        self.seedName = seed
        self.seedModel = GetSeed(seed)
        self.testKitGenerator = TestKitGenerator(seed)
        filtor.tkg = self.testKitGenerator

        self.n = n
        self.maxEachLayer = maxEachLayer

        return

    def startAGen(self, passedLastGenTreeNodes: list[TreeNode], block, gen):
        count = 1
        currentGenTreeNodes = []
        with alive_bar(self.n * len(passedLastGenTreeNodes), bar="filling", spinner="classic", title=f"{self.seedName}-{gen}") as bar:
            for father in passedLastGenTreeNodes:
                currentGenTreeNodes += father.spawn(self.n, block, count, bar)
                count += self.n
        # currentGenTreeNodes = cut(currentGenTreeNodes, self.maxEachLayer)
        for node in currentGenTreeNodes:
            f = open(f"{node.casePath}/result.json", "r", encoding="utf-8")
            d = json.load(f)
            f.close()
            if (not d["go result"]) or (not d["train result"]):
                f = open(f"{self.baseReportPath}/{self.seedName}_{node.generation}_{node.index}.json", "w", encoding="utf-8")
                json.dump(d, f, indent=2)
                f.close()
        currentGenTreeNodes = currentGenTreeNodes[:self.maxEachLayer]
        return currentGenTreeNodes

    def start(self):
        candidateBlocks = copy.deepcopy(self.seedModel.graph)
        template = copy.deepcopy(self.seedModel)
        template.graph.clear()

        root = TreeNode()
        root.newNode(self.baseOutputPath, 0, 1, self.seedName)
        root.outputShape = inputShapeTable[self.seedName]
        root.runResult = True
        root.trainResult = True
        root.father = (-1, -1)
        root.saveCase(template)

        lastGen = [root]
        currentGen = []

        for (i, block) in enumerate(candidateBlocks):
            currentGen += self.startAGen(lastGen, block, i+1)
            lastGen = copy.deepcopy(currentGen)
            currentGen = []


if __name__ == "__main__":
    a = Assembler("lenet", 2, 500, "lt")
    self = a


    def bb():
        print("=======================")


    candidateBlocks = copy.deepcopy(self.seedModel.graph)
    template = copy.deepcopy(self.seedModel)
    template.graph.clear()

    b = a.seedModel.graph[0]

    root = TreeNode()
    root.newNode(self.baseOutputPath, 0, 1, self.seedName)
    root.outputShape = inputShapeTable[self.seedName]
    root.runResult = True
    root.trainResult = True
    root.father = (-1, -1)
    root.saveCase(template)

    childs = root.spawn(2, b, 1, bb)
    child = childs[0]
