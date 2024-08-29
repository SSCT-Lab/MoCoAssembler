LAYER = 1  # Keep params in declaration.
OP = 2  # No params need, just use it to calculate. 
PARAM_NEED_OP = 3  # An op but need param behind.


class Block:

    def __init__(self, api_name, params, node_name, input_symbols, output_symbols):
        # Block data
        self.node_name: str = node_name
        self.api_name: str = api_name
        self.params: dict = params
        self.input_symbols: list[str] = input_symbols
        self.output_symbols: list[str] = output_symbols

        # Block controller info
        self.is_child_model: bool = False
        self.child_model = None  # Type: Model
        self.block_type = LAYER
        self.dim = -1

    def set_child_model(self, model):
        self.is_child_model = True
        self.child_model = model

    def _get_param_value(self, param_name, default):
        if param_name in self.params.keys():
            return self.params[param_name]
        else:
            return default

    def generate_declaration_statement(self):
        if self.block_type == LAYER:
            return f"    {', '.join(self.output_symbols)} = {self.api_name}({self._generate_param_string()})({', '.join(self.input_symbols)})"
        elif self.block_type == OP:
            return f"    {', '.join(self.output_symbols)} = {self.api_name}({self._generate_param_string()})"
        elif self.block_type == PARAM_NEED_OP:
            return f"    {', '.join(self.output_symbols)} = {self.api_name}({self._generate_param_string()})"
        else:
            return ""

    def _generate_param_string(self):
        return ", ".join([f"{key}={self.params[key]}" for key in self.params.keys()])

    def transmit_dim(self, pre_dim):
        if "1D" in self.api_name:
            self.dim = 1
        elif "2D" in self.api_name:
            self.dim = 2
        elif "3D" in self.api_name:
            self.dim = 3
        else:
            self.dim = pre_dim

    def get_param_value(self, param, default):
        if param in self.params.keys():
            return self.params[param]
        else:
            return default

    def check_block_unknown(self):
        for param in self.params.keys():
            if self.params[param] == "UNKNOWN":
                return False
        return True
