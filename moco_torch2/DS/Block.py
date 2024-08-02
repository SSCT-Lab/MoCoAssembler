import copy


LAYER = 1  # Keep params in declaration. (self.a = torch.nn.a(var1=1, var2=2), x = self.a(x))
OP = 2  # No params need, just use it to calculate. (self.a = torch.sin, x = self.a(x))
PARAM_NEED_OP = 3  # An op but need param behind. (self.a = F.pad, x = self.a(x, var1=1, var2=2))
IN_POOL = ["in_channels", "in_features", "num_features"]
OUT_POOL = ["out_channels", "out_features"]

class Block:
    def __init__(self, apiName, params, nodeName, inputSymbols, outputSymbols):

        # Block data
        self.nodeName: str = nodeName
        self.apiName: str = apiName
        self.params: dict = params
        self.inputSymbols: list[str] = inputSymbols
        self.outputSymbols: list[str] = outputSymbols

        # Block controller info
        self.isChildModel: bool = False
        self.childModel = None  # Type: Model
        self.blockType = LAYER
        self.inChannels = 0
        self.outChannels = 0
        self.dim = 2
        return

    def SetChildModel(self, model):
        self.isChildModel = True
        self.childModel = model

    def SetBlockType(self, blockType):
        self.blockType = blockType
        assert self.blockType in [LAYER, OP, PARAM_NEED_OP]

    def GetParamValue(self, paramName, default):
        if paramName in self.params.keys():
            return self.params[paramName]
        else:
            return default

    def GenerateDeclarationStatement(self):
        if self.blockType == LAYER:
            return f"        self.{self.nodeName} = {self.apiName}({self.GenerateParamString()})"
        elif self.blockType == OP:
            return f"        self.{self.nodeName} = {self.apiName}"
        elif self.blockType == PARAM_NEED_OP:
            return f"        self.{self.nodeName} = {self.apiName}"
        else:
            return ""

    def GenerateForwardStatement(self):
        if self.blockType == LAYER:
            return f"        {', '.join(self.outputSymbols)} = self.{self.nodeName}({', '.join(self.inputSymbols)})"
        elif self.blockType == OP:
            return f"        {', '.join(self.outputSymbols)} = self.{self.nodeName}({', '.join(self.inputSymbols)})"
        elif self.blockType == PARAM_NEED_OP:
            return f"        {', '.join(self.outputSymbols)} = self.{self.nodeName}({', '.join(self.inputSymbols)}, {self.GenerateParamString()})"
        else:
            return ""

    def GeneratePreCheckStatement(self):
        if self.blockType == LAYER:
            return f"    p = {self.apiName}({self.GenerateParamString()})\n" \
                   f"    y = p(x).shape\n"
        elif self.blockType == OP:
            return f"    p = {self.apiName}\n" \
                   f"    y = p(x).shape\n"
        elif self.blockType == PARAM_NEED_OP:
            return f"    p = {self.apiName}\n" \
                   f"    y = p(x, {self.GenerateParamString()}).shape\n"
        else:
            return ""

    def GenerateChildModelDeclaration(self):
        return "" if not self.isChildModel else self.childModel.AssembleModel()

    def GenerateParamString(self):
        return ", ".join([f"{key}={self.params[key] if not isinstance(self.params[key], str) else self.parenStr(self.params[key])}" for key in self.params.keys()])

    def parenStr(self, s):
        return "'" + s + "'"

    def SetShape(self, inC=0, outC=0, d=0):
        if inC > 0:
            self.inChannels = inC
        if outC > 0:
            self.outChannels = outC
        if d > 0:
            self.dim = d

    def FixShape(self):
        # This function should not be used unless after mutation.
        for paramName in self.params.keys():
            if paramName in IN_POOL:
                self.params[paramName] = self.inChannels
            if paramName in OUT_POOL:
                self.params[paramName] = self.outChannels

    def InitializeShape(self, pre=None):
        assert isinstance(pre, Block) or (pre is None)
        inFlag = None
        for _in in IN_POOL:
            if _in in self.params.keys():
                inFlag = _in
        outFlag = None
        for out in OUT_POOL:
            if out in self.params.keys():
                outFlag = out
        if pre is None:
            if "1d" in self.apiName:
                self.dim = 1
            else:
                self.dim = 2
            self.inChannels = self.GetParamValue(inFlag, self.inChannels)
            if not outFlag:
                self.outChannels = self.inChannels
            else:
                self.outChannels = self.GetParamValue(outFlag, self.outChannels)
            return
        else:
            if inFlag:
                self.inChannels = self.GetParamValue(inFlag, self.inChannels)
            else:
                self.inChannels = pre.outChannels
            self.dim = pre.dim
            if not outFlag:
                self.outChannels = self.inChannels
            else:
                self.outChannels = self.GetParamValue(outFlag, self.outChannels)
            return

    def ChildModelAssign(self):
        # This function should be executed after CalculateShape
        if not self.isChildModel:
            return
        paramValueTable = self.params
        childModel = copy.deepcopy(self.childModel)
        for block in childModel.graph:
            childParams = block.params
            for key in childParams.keys():
                value = childParams[key]
                if isinstance(value, str) and value in paramValueTable.keys():
                    childParams[key] = paramValueTable[value]
            block.params = childParams
        self.childModel = childModel
        self.childModel.CalculateShape()
        self.apiName = self.apiName + str(hash(childModel))
        self.childModel.modelName = self.apiName
        self.params.clear()
        return

    def ChildModelInitializeShape(self):
        # This function should be executed after CalculateShape & ChildModelAssign
        if not self.isChildModel:
            return
        self.childModel.CalculateShapeChildModel(self.inChannels, self.outChannels)

    def InferType(self):
        if "functional" in self.apiName:
            if len(self.params) < 1:
                self.blockType = OP
            else:
                self.blockType = PARAM_NEED_OP
        else:
            self.blockType = LAYER
