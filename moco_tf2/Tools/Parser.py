import re

from DS.Block import Block
from DS.Model import Model


def get_seed(seed_name):
    parser = Parser(file_path=f"Data/seed_models/{seed_name}.py")
    model = parser.parse(seed_name)
    return model


class Parser:
    def __init__(self, content="", file_path=""):
        self.content = content
        if content == "":
            with open(file_path, "r", encoding="utf-8") as f:
                self.content = f.read()

        self.line_lexers: list[str] = self.content.split("\n")
        self.look_ahead: int = 0

        self.child_models: dict[str: Model] = {}

        self.graph: list[Block] = []
        self.model_inputs = []
        self.model_outputs = []
        self._skip_layer = ['tf.keras.Input', 'tf.keras.models.Model']

    def has_next_line(self):
        return self.look_ahead < len(self.line_lexers) - 1

    def parse(self, model_name):
        buffer = []
        while self.has_next_line():
            current_line = self.line_lexers[self.look_ahead]
            self.look_ahead += 1
            if f"def {model_name}" in current_line:
                self.model_inputs = [var.strip() for var in
                                    current_line[current_line.find('(') + 1:current_line.find(')')].split(',')]
                # 主模块
                while self.has_next_line():
                    current_line = self.line_lexers[self.look_ahead]
                    self.look_ahead += 1
                    if self._skip_layer[0] in current_line or self._skip_layer[1] in current_line or current_line == '':
                        continue
                    if "return " in current_line:
                        self.graph = self.__parse(buffer)
                        buffer.clear()
                        break
                    buffer.append(current_line)
            elif "def inception" in current_line:
                # 自定义模块
                child_name = re.findall(r"def (inception.*?)\(.*?", current_line, re.S)[0]
                child_inputs = [var.strip() for var in
                                    current_line[current_line.find('(') + 1:current_line.find(')')].split(',')]
                while self.has_next_line():
                    current_line = self.line_lexers[self.look_ahead]
                    self.look_ahead += 1
                    if "return " in current_line:
                        chile_graph = self.__parse(buffer)
                        child_outputs = chile_graph[-1].output_symbols
                        chile_model = Model(chile_graph, child_inputs, child_outputs)
                        chile_model.model_name = child_name
                        self.child_models[child_name] = chile_model
                        buffer.clear()
                        break
                    buffer.append(current_line)
            else:
                continue

        self.model_outputs = self.graph[-1].output_symbols
        model = Model(self.graph, self.model_inputs, self.model_outputs)
        model.model_name = model_name

        for _block in self.graph:
            if _block.is_child_model:
                _block.set_child_model(self.child_models[_block.api_name])
                _block.api_name = _block.api_name + str(hash(_block.api_name))

        return model

    def __parse(self, declarations:[list]):
        main_blocks = []
        for declaration in declarations:
            declaration = declaration.strip()
            if declaration == "": continue
            layer_infos = re.match(r"(.*?)\s=\s(.+?)\((.*?)\)\((.*?)\)", declaration, re.S)
            ops_infos = re.match(r"(.*?)\s=\s(.+?)\((.*?)\)", declaration, re.S)
            if not layer_infos and not ops_infos:
                print(f"Invalid model definition format, declarationStatement: {declaration}")
                break

            if layer_infos:
                api_name = layer_infos.group(2)
                params = self.__parse_params(layer_infos.group(3), False)
                node_name = layer_infos.group(1)
                input_symbols = layer_infos.group(4)
                output_symbols = layer_infos.group(1)

                block = Block(api_name, params, node_name, [input_symbols], [output_symbols])
                block.block_type = 1
            else:
                input_symbols, params = self.__parse_params(ops_infos.group(3), True)
                api_name = ops_infos.group(2)
                node_name = ops_infos.group(1)
                output_symbols = ops_infos.group(1)

                block = Block(api_name, params, node_name, [input_symbols], [output_symbols])
                block.block_type = 2 if len(params) == 0 else 3

            if "inception" in block.api_name:
                block.is_child_model = True

            main_blocks.append(block)
        return main_blocks

    def __parse_params(self, params, is_ops=False):
        params_info = re.findall(r".*?\((.*?)\).*?", params, re.S)
        for _ in params_info:
            __ = _.replace(" ", "")
            params = params.replace(_, __)
        params_info2 = re.findall(r".*?\[(.*?)].*?", params, re.S)
        for _ in params_info2:
            __ = _.replace(" ", "")
            params = params.replace(_, __)

        params = params + ', '
        details = re.findall(r"(?P<param>.*?)=(?P<value>.*?), ", params, re.S)

        if is_ops:
            # 算子类型的API
            return details[0][1], {key: value for key, value in details}
        else:
            return {key: value for key, value in details}
