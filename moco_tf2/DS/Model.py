import os
import random

from DS import Block
from Utils import utils


MATH_PATH = "/Users/wuduo/Documents/BioWork/MoCo/MOCO-3/MoCoAssembler/moco_tf2/Data/tf_layer_infos/math"


class Model:
    def __init__(self, graph, model_inputs, model_outputs):
        self.model_name = ""
        self.graph: list[Block] = graph
        self.model_inputs: list[str] = model_inputs
        self.model_outputs: list[str] = model_outputs

    def assemble_child_model(self):
        child_code = ""
        for block in self.graph:
            if block.api_name in ["tf.keras.layers.concatenate", "tf.keras.layers.add"]:
                try:
                    _input = block.input_symbols[0][1:-1]
                    input_list = _input.split(',')
                    child_code += f"    target_height = inputs.shape[1]\n" \
                                  f"    target_width = inputs.shape[2]\n"
                    for inp in input_list:
                        child_code += f"    {inp.strip()} = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))({inp.strip()})\n"
                except: pass
            child_code += f"{block.generate_declaration_statement()}\n"

        return f"def {self.model_name}({','.join(self.model_inputs)}):\n" \
               f"{child_code}\n" \
               f"    return {','.join(self.model_outputs)}\n"

    def assemble_file(self, output_path, file_name, has_go_code=True, has_train_code=False, use_gpu=False):
        child_done = []
        model_inputs = self.model_inputs[0]
        model_outputs = self.model_outputs[0]

        main_model_code = f"def {self.model_name}({model_inputs}):\n"
        main_model_code += f"    {' = '.join(self.graph[0].input_symbols)} = tf.keras.Input(shape={model_inputs})\n"
        main_model_code += "\n".join([_.generate_declaration_statement() for _ in self.graph])
        main_model_code += f"\n" \
                           f"    model = tf.keras.models.Model(inputs={self.graph[0].input_symbols[0]}, outputs={model_outputs})\n" \
                           f"    return model\n"

        child_model_code = "\n"
        for block in self.graph:
            if block.is_child_model:
                if block.api_name not in child_done:
                    child_done.append(block.api_name)
                    extra_model_inputs = block.api_name
                    child_model_code += block.child_model.assemble_child_model(extra_model_inputs)
                    child_model_code += "\n"

        whole_model_code = f"import copy\nimport numpy as np\nimport tensorflow as tf\n" \
                    f"\n" \
                    f"\n" \
                    f"{main_model_code}\n" \
                    f"{child_model_code}\n" \
                    f"\n{self.generate_go_code(use_gpu) if has_go_code else ''}\n" \
                    f"\n{self.generate_train_code() if has_train_code else ''}\n"
        f = open(f"{output_path}/{file_name}.py", "w", encoding="utf-8")
        f.write(whole_model_code)
        f.close()
        return f"{output_path}/{file_name}.py"

    def generate_go_code(self, use_gpu=False):
        math_ops = utils.generate_input()
        device = f"with tf.device('{'/GPU:0' if use_gpu else '/CPU:0'}'):\n"
        return f"def go():\n" \
               f"    {device}" \
               f"       tf_input = tf.random.normal({utils.get_input_shape_str(self.model_name)})\n" \
               f"       tf_input = {math_ops}(tf_input)\n" \
               f"       tf_model = {self.model_name}(tf_input.shape[1:])\n" \
               f"       tf_output = tf_model(tf_input)\n" \
               f"       return tf_output\n\n\n"

    def generate_train_code(self):
        math_ops = utils.generate_input()
        return f"def chebyshev_distance(A: np.ndarray, B: np.ndarray):\n" \
               f"    if A is None or B is None:\n" \
               f"        return 0.0\n" \
               f"    if A.shape != B.shape:\n" \
               f"        return 9999999\n" \
               f"    else:\n" \
               f"        return float(np.max(np.abs(A - B)))\n" \
               f"\n" \
               f"\n" \
               f"def train(inp, label):\n" \
               f"    inp = {math_ops}(inp)\n" \
               f"    flag = True\n" \
               f"    label = tf.convert_to_tensor(label)\n" \
               f"    model_g = {self.model_name}(inp.shape[1:])\n" \
               f"    with tf.device('GPU'):\n" \
               f"        with tf.GradientTape() as tape:\n" \
               f"            output_g = model_g(inp)\n" \
               f"            loss_g = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_g)\n" \
               f"        gradients_g = tape.gradient(loss_g, model_g.trainable_variables)\n" \
               f"        gradients_dic_g = {{}}\n" \
               f"        for var, gradient in zip(model_g.trainable_variables, gradients_g):\n" \
               f"            if gradient != None:\n" \
               f"                gradients_dic_g.setdefault(var.name.replace('/', '.')[:-2], gradient)\n" \
               f"\n" \
               f"    model_c = copy.deepcopy(model_g)\n" \
               f"    with tf.device('CPU'):\n" \
               f"        with tf.GradientTape() as tape:\n" \
               f"            output_c = model_c(inp)\n" \
               f"            loss_c = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_c)\n" \
               f"        gradients_c = tape.gradient(loss_c, model_c.trainable_variables)\n" \
               f"        gradients_dic_c = {{}}\n" \
               f"        for var, gradient in zip(model_c.trainable_variables, gradients_c):\n" \
               f"            if gradient != None:\n" \
               f"                gradients_dic_c.setdefault(var.name.replace('/', '.')[:-2], gradient)\n" \
               f"    if chebyshev_distance(output_c.numpy(), output_g.numpy()) > 1.0:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Output diff too big'\n" \
               f"    if abs(loss_c - loss_g) > 0.1:\n" \
               f"        flag = False\n" \
               f"        return flag, 'Loss diff too big'\n" \
               f"    for name in gradients_dic_c.keys():\n" \
               f"        if name in gradients_dic_g.keys():\n" \
               f"            if chebyshev_distance(gradients_dic_c[name], gradients_dic_g[name]) > 0.1:\n" \
               f"                flag = False\n" \
               f"                return flag, 'Grad diff too big'\n" \
               f"    for name in gradients_dic_g.keys():\n" \
               f"        if name in gradients_dic_c.keys():\n" \
               f"            if chebyshev_distance(gradients_dic_g[name], gradients_dic_c[name]) > 0.1:\n" \
               f"                flag = False\n" \
               f"                return flag, 'Grad diff too big'\n" \
               f"    return flag, ''\n"
