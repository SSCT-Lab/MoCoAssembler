from DS import Model, RNNModel
import re


def GetSeed(seedName):
    if seedName == "LSTM":
        return RNNModel.GetLSTM()
    p = Parser(filePath=f"./Data/Torch/seed_models/{seedName}.py")
    a = p.Parse(seedName)
    a.ModelPreHandle()
    return a


class Parser:
    def __init__(self, content="", filePath=""):
        self.content: str = content
        if content == "":
            f = open(filePath, "r", encoding="utf-8")
            self.content = f.read()
            f.close()
        self.lineLexers: list[str] = self.content.split("\n")
        self.lookAhead: int = 0
        self.symbolTable: dict[str: Model.Block] = {}

        self.in_class = False
        self.in_init = False
        self.in_forward = False

        self.childModels: dict[str: Model.Model] = {}

        self.graph: list[Model.Block] = []
        self.modelInputs = []
        self.modelOutputs = []

        self.mainModelLoaded = False
        # self.resultModel = Model.Model(self.graph, self.modelInputs, self.modelOutputs)
        return

    def HasNextLine(self):
        return self.lookAhead < len(self.lineLexers) - 1

    def NextLine(self):
        self.lookAhead += 1

    def CurrentLine(self):
        return self.lineLexers[self.lookAhead]

    def Parse(self, modelName=""):
        buffer = []
        while self.HasNextLine():
            currentLine = self.CurrentLine()
            self.NextLine()
            if not self.in_class:
                if 'class' in currentLine and 'Module' in currentLine:
                    if not self.mainModelLoaded:
                        self.in_class = True
                        self.symbolTable.clear()
                        self.mainModelLoaded = True
                        continue
                    else:
                        buffer.clear()
                        buffer.append(currentLine)
                        childModelName = currentLine.split("class ", 1)[1].split("(", 1)[0]
                        while self.HasNextLine() and "return" not in self.CurrentLine():
                            buffer.append(self.CurrentLine())
                            self.NextLine()
                        buffer.append(self.CurrentLine())
                        if "return" in buffer[-1]:
                            childModel = Parser(content=("\n\n" + "\n".join(buffer)) + "\n\n").Parse(childModelName)
                            self.childModels[childModelName] = childModel
                        self.in_class = False
                        buffer.clear()
                else:
                    continue
            else:
                if '__init__' in currentLine:
                    self.in_init = True
                    self.in_forward = False
                    buffer.clear()
                elif 'def forward' in currentLine:
                    self.ParseDeclaration(buffer)
                    buffer.clear()
                    self.in_init = False
                    self.in_forward = True
                    model_inputs = [var.strip() for var in currentLine[currentLine.find('(') + 1:currentLine.find(')')].split(',')]
                    model_inputs.remove('self')  # 移除 self
                    self.modelInputs = model_inputs
                    continue
                elif 'return' in currentLine:
                    self.ParseForward(buffer)
                    buffer.clear()
                    model_outputs = [var.strip() for var in currentLine.replace('return', '').split(',')]
                    self.modelOutputs = model_outputs
                    self.in_class = False
                    self.in_init = False
                    self.in_forward = False
                else:
                    if self.in_init or self.in_forward:
                        buffer.append(currentLine)
                    else:
                        continue
        model = Model.Model(self.graph, self.modelInputs, self.modelOutputs)
        model.modelName = modelName

        if "Inception" in self.childModels.keys():
            inception = self.childModels["Inception"]

            for _block in self.graph:
                if _block.apiName == "Inception":
                    _block.SetChildModel(inception)
        return model

    def ParseDeclarationStatement(self, declarationStatement: str):
        if declarationStatement == "":
            return
        pattern = r"self\.(\w+)\s*=\s*(torch\.nn\.\w+|Inception)\((.*?)\)$"
        match = re.match(pattern, declarationStatement.strip())
        pattern2 = r"self\.(\w+)\s*=\s*(torch\.\w+)"
        match2 = re.match(pattern2, declarationStatement.strip())
        if (not match) and (not match2):
            print(f"Invalid model definition format: {declarationStatement}")
            return
        if match:
            nodeName = match.group(1)
            apiName = match.group(2)
            params_string = match.group(3)
            if params_string is None:
                noParamFlag = True
                params = {}
            else:
                noParamFlag = False
                params = self.__parse_params(params_string)
            block = Model.Block(apiName, params, nodeName, ["TBD"], ["TBD"])
            if noParamFlag:
                block.blockType = 2
        elif match2:
            nodeName = match2.group(1)
            apiName = match2.group(2)
            block = Model.Block(apiName, {}, nodeName, ["TBD"], ["TBD"])
            block.blockType = 2
        else:
            return
        self.symbolTable[nodeName] = block
        return

    def ParseDeclaration(self, declarationLineLexer: list[str]):
        for lexer in declarationLineLexer:
            self.ParseDeclarationStatement(lexer)
        return

    def ParseForwardStatement(self, forwardStatement: str):
        if forwardStatement == "":
            return
        pattern = r"^(.*?)\s*=\s*self\.(\w+)\((.*?)\)$"
        match = re.match(pattern, forwardStatement.strip())
        if not match:
            print(f"Invalid model definition format: {forwardStatement}")
            return
        output_list = match.group(1).strip()
        nodeName = match.group(2).strip()
        input_list = match.group(3).strip()
        if nodeName not in self.symbolTable.keys():
            print(f"Node name '{nodeName}' not found in symbol table.")
            return
        output_symbols = [var.strip() for var in output_list.split(',') if var.strip()]
        input_symbols = self.__parse_multi_inputs(input_list)
        block = self.symbolTable[nodeName]
        block.inputSymbols = input_symbols
        block.outputSymbols = output_symbols
        if len(block.inputSymbols) > 1 and block.blockType != 1:
            block.blockType = 3
            block.inputSymbols = [block.inputSymbols[0]]
            block.params = self.__parse_params(','.join(input_symbols[1:]))
        self.graph.append(block)
        return

    def ParseForward(self, forwardLineLexer: list[str]):
        for lexer in forwardLineLexer:
            self.ParseForwardStatement(lexer)
        return


    def __parse_multi_inputs(self, vars_string):
        inputs = []
        current_input = []
        depth = 0  # Track the depth of parentheses/brackets

        for char in vars_string:
            if char in '([{':
                depth += 1
                current_input.append(char)
            elif char in ')]}':
                depth -= 1
                current_input.append(char)
            elif char == ',' and depth == 0:
                # Only split on commas that are not nested inside parentheses/brackets
                inputs.append(''.join(current_input).strip())
                current_input = []
            else:
                current_input.append(char)

        # Add the last element if it's not empty
        if current_input:
            inputs.append(''.join(current_input).strip())

        return inputs

    def __parse_params(self, params_string):
        params = {}
        buffer = ""
        depth = 0
        key = None

        # 添加一个哨兵字符，确保最后一个参数被处理
        params_string += ','

        for char in params_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1

            if char == ',' and depth == 0:
                if '=' in buffer:
                    key, value = buffer.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    params[key] = value
                buffer = ""
            else:
                buffer += char
        for key in params.keys():
            if (isinstance(params[key], str)) and params[key].replace(".", "").isdigit():
                params[key] = int(params[key]) if "." not in params[key] else float(params[key])
        return params


if __name__ == "__main__":
    models = []
    for model in ["lenet", "alexnet", "googlenet", "mobilenet", "pointnet", "squeezenet", "vgg19"]:
        models.append(GetSeed(model))
    from ConstraintChecker import CheckBlock
    r = CheckBlock(models[0].graph[0])
    # p = Parser(filePath=f"C:/projects/MoCoAssembler-MoCo_1.0/moco_torch2/Data/Torch/seed_models/{model}.py")
    # # decl = "self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))"
    # # fowd = "x = self.conv_1(x)"
    # # p.ParseDeclarationStatement(decl)
    # # b = p.symbolTable["conv_1"]
    # # p.ParseForwardStatement(fowd)
    # # n = p.graph[0]
    # a = p.Parse(model)
    # # p = Parser(" ")
    # # p.ParseDeclarationStatement("    self.cat = torch.cat")
    # a.AssembleFile(
    #     outputPath="../Test",
    #     fileName="t"
    # )
    # seeds = ["lenet", "googlenet", "squeezenet", "mobilenet", "vgg19", "pointnet", "alexnet"]
    # for seed in seeds:
    #     a = GetSeed(seed)
    #     a.AssembleFile(
    #         "../Test",
    #         seed + "_t"
    #     )
    a = models[0]
    aa = models[2]
    aaa = models[4]
    a.AssembleFile(
        "../Test",
        "lenet_train_test",
        True,
        True,
        True
    )
    aa.AssembleFile(
        "../Test",
        "googlenet_train_test",
        True,
        True,
        True
    )
    aaa.AssembleFile(
        "../Test",
        "pointnet_train_test",
        True,
        True,
        True
    )
