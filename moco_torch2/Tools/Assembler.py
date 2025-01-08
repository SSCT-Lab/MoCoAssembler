import copy
import pickle
import os
import time
import json
import importlib.util
import sys
import traceback
import numpy as np
import re

from alive_progress import alive_bar
from collections import defaultdict
from DS.Model import Model
from DS.Block import Block
from Tools.Parser import GetSeed
from Tools.TesterKitGenerator import TestKitGenerator
from Tools.Mutator import Mutator

threshold = 0.1
mutator = Mutator()
vari = []


class Filter:
    def __init__(self):
        self.info_lis = []
        self.tkg = None
        self.addkey("must be divisible by groups")
        self.addkey(r"missing (\d+) required positional argument(s)?")
        self.addkey("out_channels must be divisible by groups")
        self.addkey("but got Tensor of dimension")
        self.addkey("mat1 and mat2 shapes cannot be multiplied")
        self.addkey("but got input of size")
        self.addkey("Output size is too small")
        self.addkey("expected to be in range of")
        self.addkey("channels instead")
        self.addkey("float() argument must be a string or a real number, not ")
        self.addkey("flatten() has invalid args: start_dim cannot come after end_dim")
        self.addkey(r"Only (\d+D,?\s?)+padding with non-constant padding are supported for now")
        self.addkey("'tuple' object has no attribute")
        self.addkey(r"Expected (\d+)D or (\d+)D \(batch mode\) tensor for input")
        self.addkey(r"pool(\d+)d\(\): Expected")
        self.addkey("tensor expected for input")
        self.addkey("while checking arguments for")
        self.addkey(r"expects input with > (\d+) dims")
        self.addkey("must be tuple of ints, but found")
        self.addkey("Sizes of tensors must match")
        self.addkey("Tensors must have same number of dimensions")
        self.addkey("must be Tensor, not tuple")
        self.addkey("input has inconsistent input_size")
        self.addkey("input.size(-1) must be equal to input_size")
        self.addkey(r"It is expected dilation equals to (\d+)")
        self.addkey(r"Input dimension should be at least (\d+)")
        self.addkey("running_mean should contain")
        self.addkey("weight should contain")
        self.addkey("Expected weight to be")
        self.addkey(r"Padding length must be divisible by (\d+)")
        self.addkey(r"It is expected stride equals to (\d+)")
        self.addkey(r"expected (\d+)D input")
        self.addkey("Expected size of input")
        self.addkey("The size of tensor a")
        self.addkey("Kernel size can't be greater than actual input size")
        self.addkey("object has no attribute")

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if len(re.findall(s, string)) > 0:
                return False
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
    "resnet18": [1, 3, 224, 224],
    "vgg19": [1, 3, 224, 224],
    "squeezenet": [1, 3, 224, 224],
    "pointnet": [2, 3, 2048],
    "LSTM": [1, 3, 2048]
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
        return 9999999
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
        x, _, s1 = filtor.tkg.generate_kit()
        xt, yt, s2 = filtor.tkg.generate_kit()
        filePath = f"{self.casePath}/{self.seedName}_{self.generation}_{self.index}_train.py"
        sys.path.append(self.casePath)
        
        if len(self.outputShape) >= 2:
            inChannels = 1
            for i in range(1, len(self.outputShape)):
                inChannels *= self.outputShape[i]
        elif len(self.outputShape) == 1:
            inChannels = self.outputShape[0]
        else:
            inChannels = 1
        if inChannels > 20000:
            return 1.0, "", (-2, -2)

        start = time.time()
        try:
            module_name = f"{self.seedName}_{self.generation}_{self.index}_go"
            spec = importlib.util.spec_from_file_location(module_name, filePath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            result, info = module.train(x + 0.00001, xt + 0.00001, yt)
            error_message = info
            end = time.time()
        except Exception as e:
            result, info = False, ""
            end = start - 1
            error_message = traceback.format_exc()

        sys.path.remove(self.casePath)
        trainTime = end - start

        if not result:
            trainTime = -1.0

        return trainTime, error_message, (s1, s2)

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

            # vari
            if mutateInfo not in vari:
                vari.append(mutateInfo)

            if inChannels != -1:
                if not (self.seedName == "LSTM" and self.generation == 0):
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
            # brandNewModel.modelOutputs = mutatedBlock.outputSymbols

            brandNewModel.extraOp = mutator.GenerateRandomOp()

            child.saveCase(brandNewModel)
            child.mutateInfo = mutateInfo

            runTime, runErrorInfo = child.run()
            runResult = (runTime >= 0.0)
            if runResult:
                trainTime, trainErrorInfo, caseIndex = child.train()
                trainResult = (trainTime >= 0.0)
            else:
                trainTime, trainErrorInfo, trainResult = -1.0, "", False
                caseIndex = (-1, -1)
            result = {
                "go result": runResult,
                "go time": runTime,
                "go error info": runErrorInfo,
                "train result": trainResult,
                "train time": trainTime,
                "train error info": trainErrorInfo,
                "father": str(self.no()),
                "mutate info": mutateInfo,
                "case": str(caseIndex)
            }
            f = open(f"{child.casePath}/result.json", "w", encoding="utf-8")
            json.dump(result, f, indent=2)
            f.close()

            if runResult and trainResult:
                child.weight = runTime + trainTime
                res.append(child)

        return res  # Here, model of these TreeNodes has been saved, and it just return fine nodes.

    def spawn2(self, block: Block):
        model = self.getModel()
        newBlocks, tags = mutator.BoundaryMutate(block)
        total = len(newBlocks)
        with alive_bar(total, bar="filling", spinner="classic", title=f"{self.seedName}-{self.generation}-b") as bar:
            for i in range(total):
                bar()
                newBlock = newBlocks[i]
                if not self.preCheck(newBlock):
                    continue
                tag = tags[i]
                newModel = copy.deepcopy(model)
                newModel.graph.append(newBlock)
                newChild = TreeNode()
                newChild.newNode(self.basePath, self.generation, i+1, self.seedName)
                newChild.saveCase(newModel)
                expect = "BELOW" in tag
                runTime, runErrorInfo = newChild.run()
                runResult = (runTime >= 0.0)
                result = {
                    "go result": runResult,
                    "go time": runTime,
                    "go error info": runErrorInfo,
                    "father": str(self.no()),
                    "tag": str(tag)
                }
                f = open(f"{newChild.casePath}/result.json", "w", encoding="utf-8")
                json.dump(result, f, indent=2)
                f.close()
                if not runResult:
                    if not expect:
                        f = open(f"{newChild.casePath}/report", "w", encoding="utf-8")
                        f.close()


def cut(nodes, limit):
    # Step 1: 分桶
    buckets = defaultdict(list)
    for node in nodes:
        buckets[node.mutateInfo].append(node)

    # Step 2: 桶内排序
    for bucket in buckets.values():
        bucket.sort(key=lambda x: x.weight)

    # Step 3: 截断桶后50%
    truncated = []
    for bucket in buckets.values():
        cutoff = len(bucket) // 2 + 1
        truncated.extend(bucket[:cutoff])  # 取前50%加入到truncated列表

    # Step 4: 检查limit
    if len(truncated) > limit:
        # 如果超过limit，从每个桶中取最小的，直到达到limit
        result = []
        while len(result) < limit:
            # 按照每个桶的最小元素（已排序）循环加入
            for bucket in buckets.values():
                if bucket and len(result) < limit:
                    result.append(bucket.pop(0))  # 弹出每个桶的第一个元素
        return result
    else:
        # 如果不超过limit，直接返回truncated
        return truncated


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

    def getDefaultModel(self, gen: int, count: int):
        defaultModel = copy.deepcopy(self.seedModel)
        defaultModel.graph = defaultModel.graph[:gen]
        res = []

        for c in range(count):
            node = TreeNode()
            node.newNode(self.baseOutputPath, gen, self.maxEachLayer + c + 1, self.seedName)
            node.saveCase(defaultModel)
            node.run()
            node.father = (-1, -1)
            res.append(copy.deepcopy(node))

        return res

    def startAGen(self, passedLastGenTreeNodes: list[TreeNode], block, gen):
        vari.clear()
        count = 1
        currentGenTreeNodes = []
        with alive_bar(self.n * len(passedLastGenTreeNodes), bar="filling", spinner="classic", title=f"{self.seedName}-{gen}") as bar:
            for father in passedLastGenTreeNodes:
                newNodes = father.spawn(self.n, block, count, bar)
                currentGenTreeNodes += newNodes
                count += self.n
        currentGenTreeNodes = cut(currentGenTreeNodes, self.maxEachLayer)
        # for node in currentGenTreeNodes:
        #     f = open(f"{node.casePath}/result.json", "r", encoding="utf-8")
        #     d = json.load(f)
        #     f.close()
        #     if (not d["go result"]) or (not d["train result"]):
        #         f = open(f"{self.baseReportPath}/{self.seedName}_{node.generation}_{node.index}.json", "w", encoding="utf-8")
        #         json.dump(d, f, indent=2)
        #         f.close()
        # currentGenTreeNodes = currentGenTreeNodes[:self.maxEachLayer]
        if len(currentGenTreeNodes) < 1:
            print(f"fix {len(vari)} default models in gen {gen}")
            currentGenTreeNodes = self.getDefaultModel(gen, len(vari))
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

        print("result scanning")
        for gen in os.listdir(self.baseOutputPath):
            for case in os.listdir(f"{self.baseOutputPath}/{gen}"):
                casePath = f"{self.baseOutputPath}/{gen}/{case}"
                targetName = f"{self.seedName}-{gen}-{case}"
                if "result.json" in os.listdir(casePath):
                    f = open(f"{casePath}/result.json", "r", encoding="utf-8")
                    d = json.load(f)
                    f.close()
                    if (not d["go result"]) or (not d["train result"]):
                        f = open(f"{self.baseReportPath}/{targetName}.json", "w", encoding="utf-8")
                        json.dump(d, f, indent=2)
                        f.close()
        print("result scanning finished")

        return


class Bssembler:
    def __init__(self, seed, experimentName=None, outputPath="./Output"):
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
        filtor.tkg = None

        return

    def startAGen(self, template, block, gen):
        father = TreeNode()
        father.newNode(self.baseOutputPath, gen, 0, self.seedName)
        father.saveCase(template)
        father.run()
        father.spawn2(block)
        return

    def start(self):
        candidateBlocks = copy.deepcopy(self.seedModel.graph)
        template = copy.deepcopy(self.seedModel)
        template.graph.clear()

        for (i, block) in enumerate(candidateBlocks):
            template2 = copy.deepcopy(template)
            self.startAGen(template2, block, i+1)
            template.graph.append(block)


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
