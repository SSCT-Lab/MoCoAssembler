import copy
import os
import random

from DS.Block import Block, LAYER, OP, PARAM_NEED_OP, IN_POOL, OUT_POOL
import json
import yaml
from Tools.ConstraintChecker import CheckBlock

dataPath = f"./Data/Jittor/jittor_layer_info"
threshold = 0.3
__DTYPE = ["int", "float", "string", "boolean"]
__STRUCTURE = ["scalar", "tuple", "list"]
# Mutate mode
MIN_MODE = "min"
NORMAL_MODE = "normal"
MAX_MODE = "max"
MUTATE_MODES = [NORMAL_MODE, MIN_MODE, NORMAL_MODE, MAX_MODE, NORMAL_MODE]


def GetApiList():
    fileList = os.listdir(dataPath)
    aL = []
    for file in fileList:
        aL.append(file.replace("_", ".").replace(".json", ""))
    return aL


apiList = GetApiList()


def CheckInfo():
    for api in apiList:
        d = LoadInfo(api)
        defaultEntranceFlag = False
        for key in d.keys():
            pD = d[key]
            if "dtype" not in pD.keys() or \
                    "range" not in pD.keys() or \
                    "structure" not in pD.keys() or \
                    "shape" not in pD.keys() or \
                    "default" not in pD.keys():
                print(f"{api}- param {key} has some problem: lack key in dict.")
                break

            # check dtype
            typeList = pD["dtype"]
            if not isinstance(typeList, list):
                print(f"{api}- param {key} has some problem: dtype is not a list.")
                break
            pFlag = False
            for tp in typeList:
                if tp not in ["int", "float", "string", "boolean"]:
                    pFlag = True
                    break
            if pFlag:
                print(f"{api}- param {key} has some problem: wrong type in dtype list.")
                break

            # check range
            rangeList = pD["range"]
            if not isinstance(rangeList, list):
                print(f"{api}- param {key} has some problem: range is not a list.")
                break
            if len(rangeList) != len(typeList):
                print(f"{api}- param {key} has some problem: lengths of range and dtype don't equal.")
                break
            pFlag = False
            for rg in rangeList:
                if not isinstance(rg, list) and not (rg is None):
                    pFlag = True
                    break
            if pFlag:
                print(f"{api}- param {key} has some problem: wrong type in range list.")
                break

            # check structure
            structureList = pD["structure"]
            if not isinstance(structureList, list):
                print(f"{api}- param {key} has some problem: structure is not a list.")
                break
            pFlag = False
            for tp in structureList:
                if tp not in ["scalar", "tuple", "list"]:
                    pFlag = True
                    break
            if pFlag:
                print(f"{api}- param {key} has some problem: wrong type in structure list.")
                break

            # check shape
            shapeList = pD["shape"]
            if not isinstance(shapeList, list):
                print(f"{api}- param {key} has some problem: shape is not a list.")
                break
            if len(shapeList) != len(structureList):
                print(f"{api}- param {key} has some problem: lengths of structure and shape don't equal.")
                break
            pFlag = False
            for sp in shapeList:
                if not isinstance(sp, int):
                    pFlag = True
                    break
            if pFlag:
                print(f"{api}- param {key} has some problem: wrong type in shape list.")
                break

            # check default
            defaultValue = pD["default"]
            if not isinstance(defaultValue, int) and not isinstance(defaultValue, float) and not isinstance(
                    defaultValue, str) and defaultValue is not None:
                print(f"{api}- param {key} has some problem: default has a wrong type.")
                break
            if defaultValue is None:
                if defaultEntranceFlag:
                    print(f"{api}- param {key} has some problem: required param locates after optional param.")
                    break
            else:
                defaultEntranceFlag = True


def LoadInfo(apiName: str):
    path = f"{dataPath}/{apiName.replace('.', '_')}.json"
    f = open(path, "r", encoding="utf-8")
    d = json.load(f)
    f.close()
    return d["params"]


def LoadSimilarity(apiName: str):
    path = f"{dataPath}/../jittor_layer_similarity/{apiName}.yaml"
    path = path.replace("Conv1d.sp", "Conv1d_sp").replace("Leaky.relu", "Leaky_relu")
    f = open(path, "r", encoding="utf-8")
    d = yaml.full_load(f)
    f.close()
    return d


def GetSingleValue(dtype, range_, mode=NORMAL_MODE):
    try:
        if dtype == "int":
            _min, _max = range_
            if mode == MAX_MODE:
                return _max
            elif mode == MIN_MODE:
                return _min
            else:
                return random.randint(_min, _max)
        elif dtype == "float":
            _min, _max = range_
            if mode == MAX_MODE:
                return _max
            elif mode == MIN_MODE:
                return _min
            else:
                return random.uniform(_min, _max)
        elif dtype == "boolean":
            return random.choice([True, False])
        elif dtype == "string":
            return random.choice(range_)
        else:
            return 1
    except Exception:
        return 1


def GetRandomValue(dtype, range_, structure, shape):
    # return value and mode
    if dtype == "string" or dtype == "boolean":
        structure = "scalar"
    try:
        if structure == "scalar":
            mode = random.choice(MUTATE_MODES)
            value = GetSingleValue(dtype, range_, mode)
            valueInfo = f"{dtype}, scalar, {mode if dtype in ['int', 'float'] else value}"
            return value, valueInfo
        elif structure == "tuple" or structure == "list":
            res = []
            mode = []
            for i in range(shape):
                mode.append(random.choice(MUTATE_MODES))
            for i in range(shape):
                res.append(GetSingleValue(dtype, range_))
            return tuple(res) if structure == "tuple" else res, f"{dtype}, tuple, {str(mode)}"
        else:
            return 1, "unknown"
    except Exception:
        return 1, "unknown"


class Mutator:
    def __init__(self):
        self.apiList = GetApiList()
        self.apiInfo: dict = {}
        self.apiSimilarity: dict = {}
        for api in self.apiList:
            self.apiInfo[api] = LoadInfo(api)
            pSim = LoadSimilarity(api)
            pLis = list(pSim.keys())
            for key in pLis:
                if key not in self.apiList:
                    pSim.pop(key)
            self.apiSimilarity[api] = pSim
        self.filteredApiSimilarity = {}
        for originalApiName in self.apiList:
            dimension = ''
            if '1d' in originalApiName:
                dimension = '1d'
            elif '2d' in originalApiName:
                dimension = '2d'
            elif '3d' in originalApiName:
                dimension = '3d'
            filtered_sim = {}
            for name, probability in self.apiSimilarity[originalApiName].items():
                if name == originalApiName:
                    continue
                if probability <= threshold:
                    continue
                if dimension:
                    if ('1d' in name or '2d' in name or '3d' in name) and dimension not in name:
                        continue
                filtered_sim[name] = probability
            self.filteredApiSimilarity[originalApiName] = filtered_sim
        return

    def Mutate(self, block: Block):
        if block.isChildModel:
            return self.ChildModelMutate(block)
        else:
            return self.NormalMutate(block)

    def NormalMutate(self, block: Block):
        assert not block.isChildModel
        if block.apiName not in self.apiList:
            return block, "no mutate"
        newBlock, mutateInfo = random.choice([self.ApiNameMutate, self.ApiParamMutate])(block)
        newBlock.FixShape()
        while not CheckBlock(newBlock):
            newBlock, mutateInfo = random.choice([self.ApiNameMutate, self.ApiParamMutate])(block)
            newBlock.FixShape()
        if "mutated" not in newBlock.nodeName:
            newBlock.nodeName = newBlock.nodeName + "_mutated"
        return newBlock, mutateInfo

    def ApiNameMutate(self, block: Block):
        originalApiName = block.apiName
        targetApi = self.NewRandomApiChoice(originalApiName)
        if targetApi == originalApiName or targetApi is None:
            return block, "no mutate"
        newBlock = self.CreateNewApi(targetApi, block)
        mutateInfo = f"ApiNameMutate, {targetApi}"
        return newBlock, mutateInfo

    def ApiParamMutate(self, block: Block):
        originalApiName = block.apiName
        paramSet = self.GetParamSetOfApi(originalApiName)
        if len(paramSet) < 1:
            return block, "no mutate"
        choiceParam = random.choice(paramSet)
        value, valueInfo = self.RandomValue(originalApiName, choiceParam)
        block.params[choiceParam] = value
        mutateInfo = f"ApiParaMutate, {choiceParam}, {valueInfo}"
        return block, mutateInfo

    def ChildModelMutate(self, block: Block):
        if not block.isChildModel:
            return block, "no mutate"
        newBlock = copy.deepcopy(block)
        childModel = block.childModel
        allIndex = len(childModel.graph)
        choice = random.randint(0, allIndex-1)
        choiceLayer = childModel.graph[choice]
        choiceLayerName = choiceLayer.nodeName
        newLayer, layerMutateInfo = self.NormalMutate(choiceLayer)
        newBlock.childModel.graph[choice] = newLayer
        mutateInfo = f"ChildModelMutate, {choiceLayerName}, {layerMutateInfo}"
        return newBlock, mutateInfo

    def CreateNewApi(self, targetApi, originalBlock: Block):
        # Just return new block.
        assert targetApi in self.apiList and not originalBlock.isChildModel

        nodeName = originalBlock.nodeName
        originalInputs = originalBlock.inputSymbols
        originalOutputs = originalBlock.outputSymbols
        originalParams = originalBlock.params
        originalInChannels = originalBlock.inChannels
        originalOutChannels = originalBlock.outChannels
        originalDim = originalBlock.dim

        paramRequired = self.GetRequiredParamSetOfApi(targetApi)

        newParams = {}

        for paramName in paramRequired:
            if paramName in originalParams.keys():
                newParams[paramName] = originalParams[paramName]
            else:
                newParams[paramName], _ = self.RandomValue(targetApi, paramName)

        block = Block(targetApi, newParams, nodeName, originalInputs, originalOutputs)
        block.SetShape(originalInChannels, originalOutChannels, originalDim)
        block.FixShape()
        return block

    def RandomValue(self, apiName, paramName):
        # Random a value for a param, based on param info. Return (value, mode).
        dtype, range_ = random.choice(self.GetParamDtypeAndRange(apiName, paramName))
        structure, shape = random.choice(self.GetParamStructureAndShape(apiName, paramName))
        return GetRandomValue(dtype, range_, structure, shape)

    def GetParamSetOfApi(self, apiName):
        return list(self.apiInfo[apiName].keys())

    def GetRequiredParamSetOfApi(self, apiName):
        allSet = self.GetParamSetOfApi(apiName)
        res = []
        for param in allSet:
            if self.GetParamDefault(apiName, param) is None:
                res.append(param)
        return res

    def GetParamDtypeAndRange(self, apiName, paramName):
        info = self.apiInfo[apiName][paramName]
        dtype, _range = info["dtype"], info["range"]
        res = []
        for i in range(len(dtype)):
            d = dtype[i]
            r = _range[i]
            if r is None:
                if d == "int":
                    r = [1, 8]
                elif d == "float":
                    r = [0.0, 1.0]
                elif d == "boolean":
                    r = [True, False]
                else:
                    r = []
            res.append((d, r))
        return res

    def GetParamStructureAndShape(self, apiName, paramName):
        # print(apiName, paramName)
        info = self.apiInfo[apiName][paramName]
        structure, shape = info["structure"], info["shape"]
        res = []
        for i in range(len(structure)):
            st = structure[i]
            sp = shape[i]
            res.append((st, sp))
        return res

    def GetParamDefault(self, apiName, paramName):
        info = self.apiInfo[apiName][paramName]
        default = info["default"]
        return default

    def NewRandomApiChoice(self, originalApiName):
        sim = self.filteredApiSimilarity[originalApiName]
        if len(sim) < 1:
            return originalApiName
        total = sum(sim.values())
        r = random.uniform(0, total)
        upto = 0
        for name, probability in sim.items():
            if upto + probability >= r:
                return name
            upto += probability
        return None


if __name__ == "__main__":
    # dd = LoadInfo("torch.nn.Conv1d")
    # CheckInfo()
    # dd = LoadSimilarity("torch.nn.Conv2d")
    m = Mutator()
    # res = m.GetParamDtypeAndRange("jittor.nn.Conv2d", "in_channels")
    # print(res)
    from Parser import GetSeed
    aaa = GetSeed("lenet")
    blocks = aaa.graph
    v, b, c = blocks[0], blocks[1], blocks[2]
    aa = GetSeed("googlenet")
    inc = aa.graph[11]
    models = []
    for model in ["lenet", "alexnet", "googlenet", "mobilenet", "pointnet", "squeezenet", "vgg19"]:
        models.append(GetSeed(model))
    for model in models:
        print(model.modelName)
        blocks = model.graph
        for block in blocks:
            print(block.apiName)
            for i in range(1000):
                q, qq = m.Mutate(block)
                print(qq)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
