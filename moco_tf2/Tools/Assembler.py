import copy
import pickle
import os
import time
import json
import importlib.util
import sys
import traceback

from alive_progress import alive_bar

from Tools.Filter import Filter
from DS.Block import Block
from DS.Model import Model
from Tools.Mutator import Mutator
from Tools.Parser import get_seed
from Tools.TesterKitGenerator import TestKitGenerator
from Utils import utils


class TreeNode:
    def __init__(self):
        self.seed_name = ""
        self.generation = 0
        self.index = 0
        self.father = None
        self.children = []
        self.base_path = ""
        self.case_path = ""
        self.output_shape = [0, 0, 0, 0]
        self.weight = 9999
        self.mutate_info = ""

        self.mutator = Mutator()
        self.filter = Filter()

    def new_node(self, base_path, generation, index, seed_name):
        self.seed_name = seed_name
        self.base_path = base_path
        self.generation = generation
        self.index = index
        self.case_path = f"{base_path}/{generation}/{index}"
        os.makedirs(self.case_path, exist_ok=True)

    def save_case(self, model):
        if self.case_path == "":
            return
        else:
            os.makedirs(self.case_path, exist_ok=True)
            file_path = os.path.join(self.case_path, 'model.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            return

    def _get_model(self):
        if self.case_path == "":
            return
        else:
            file_path = os.path.join(self.case_path, 'model.pkl')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No model file found in {self.case_path}")
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            return model

    def _assemble_go_file(self):
        file_path = self.case_path
        file_name = f"{self.seed_name}_{self.generation}_{self.index}_go"
        model: Model = self._get_model()
        model.assemble_file(
            output_path=file_path,
            file_name=file_name,
            has_go_code=True,
            has_train_code=False,
            use_gpu=True
        )

    def _assemble_train_file(self):
        file_path = self.case_path
        file_name = f"{self.seed_name}_{self.generation}_{self.index}_train"
        model: Model = self._get_model()
        _flatten_block: Block = Block("tf.keras.layers.Flatten", {}, "tail_flatten", model.model_outputs, ["tail_flatten"])

        _dense_params = {"units": 10 if self.seed_name in ["lenet", "pointnet"] else 1000}
        _dense_block: Block = Block("tf.keras.layers.Dense", _dense_params, "tail_fc", ["tail_flatten"], ["tail_fc"])
        model.graph.append(_flatten_block)
        model.graph.append(_dense_block)
        model.model_outputs = _dense_block.output_symbols
        model.assemble_file(
            output_path=file_path,
            file_name=file_name,
            has_go_code=True,
            has_train_code=True,
            use_gpu=True
        )

    def _pre_check(self, block: Block):
        code = f"import tensorflow as tf\n" \
               f"\n" \
               f"\n" \
               f"def pre_check():\n" \
               f"    {', '.join(block.input_symbols)} = tf.random.normal({str(self.output_shape)})\n" \
               f"{block.generate_declaration_statement()}\n"
        sys.path.append(f"{self.base_path}/../..")
        with open(f"{self.base_path}/../../pre_check.py", "w", encoding="utf-8") as f:
            f.write(code)
        try:
            module_name = "pre_check"
            spec = importlib.util.spec_from_file_location(module_name, f"{self.base_path}/../../pre_check.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            module.pre_check()
            error_message = ""
        except Exception as e:
            error_message = str(e).replace("\n", "")
        sys.path.remove(f"{self.base_path}/../..")
        if error_message == "":
            return True
        else:
            return self.filter.judge(error_message)

    def run(self):
        self._assemble_go_file()
        file_path = f"{self.case_path}/{self.seed_name}_{self.generation}_{self.index}_go.py"
        sys.path.append(self.case_path)
        start = time.time()
        try:
            module_name = f"{self.seed_name}_{self.generation}_{self.index}_go"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.output_shape = module.go().shape
            error_message = ""
            end = time.time()
        except Exception as e:
            end = start - 1
            error_message = traceback.format_exc()
        sys.path.remove(self.case_path)

        run_time = end - start
        return run_time, error_message

    def train(self):
        self._assemble_train_file()
        inp, label = TestKitGenerator(self.seed_name).generate_kit()
        file_path = f"{self.case_path}/{self.seed_name}_{self.generation}_{self.index}_train.py"
        sys.path.append(self.case_path)

        start = time.time()
        try:
            module_name = f"{self.seed_name}_{self.generation}_{self.index}_go"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            flag, error_message = module.train(inp, label)
            end = time.time()
        except Exception as e:
            flag, error_message = False, ""
            end = start - 1
            error_message = traceback.format_exc()

        sys.path.remove(self.case_path)
        train_time = end - start

        if not flag:
            train_time = -1.0

        return train_time, error_message

    def spawn_fuzzing(self, n, block, start_count, bar):
        res = []
        count = start_count - 1
        for i in range(n):
            count += 1
            child = TreeNode()
            child.new_node(self.base_path, self.generation + 1, count, self.seed_name)
            child.father = self
            self.children.append(child)

            bar()

            mutated_block, mutate_info = self.mutator.mutate(block)

            retry_count = 0
            pre_check_passed = False
            while retry_count < 9:
                retry_count += 1
                if self._pre_check(mutated_block):
                    pre_check_passed = True
                    break
                else:
                    mutated_block, mutate_info = self.mutator.mutate(block)
            if not pre_check_passed:
                count += 1
                continue

            brand_new_model = self._get_model()
            brand_new_model.graph.append(mutated_block)
            brand_new_model.model_outputs = mutated_block.output_symbols
            child.save_case(brand_new_model)
            child.mutate_info = mutate_info

            run_time, run_error_info = child.run()
            run_result = (run_time >= 0.0)
            if run_result:
                train_time, train_error_info = child.train()
                train_result = (train_time >= 0.0)
            else:
                train_time, train_error_info, train_result = -1.0, "", False
            result = {
                "go result": run_result,
                "go time": run_time,
                "go error info": run_error_info,
                "train result": train_result,
                "train time": train_time,
                "train error info": train_error_info,
                "father": f"{self.generation}, {self.index}",
                "mutate info": mutate_info
            }
            f = open(f"{child.case_path}/result.json", "w", encoding="utf-8")
            json.dump(result, f, indent=2)
            f.close()

            if run_result and train_result:
                child.weight = run_time + train_time
                res.append(child)

        return res


class Assembler:
    def __init__(self, seed, n=2, max_each_layer=500, experiment_name=None, output_path="../output"):
        if experiment_name is None:
            self.experiment_name = seed + str(time.time())
        else:
            self.experiment_name = experiment_name
        self.base_output_path = f"{output_path}/{experiment_name}/tree"
        self.base_report_path = f"{output_path}/{experiment_name}/report"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(self.base_output_path, exist_ok=True)
        os.makedirs(self.base_report_path, exist_ok=True)

        self.seed_name = seed
        self.seed_model = get_seed(seed)

        self.n = n
        self.max_each_layer = max_each_layer

    def _start_generation(self, passed_last_gen_tree_nodes: list[TreeNode], block, gen):
        count = 1
        current_gen_tree_nodes = []
        with alive_bar(self.n * len(passed_last_gen_tree_nodes), bar="filling", spinner="classic", title=f"{self.seed_name}-{gen}") as bar:
            for father in passed_last_gen_tree_nodes:
                current_gen_tree_nodes += father.spawn_fuzzing(self.n, block, count, bar)
                count += self.n
        current_gen_tree_nodes = utils.cut(current_gen_tree_nodes, self.max_each_layer)
        for node in current_gen_tree_nodes:
            f = open(f"{node.case_path}/result.json", "r", encoding="utf-8")
            d = json.load(f)
            f.close()
            if (not d["go result"]) or (not d["train result"]):
                f = open(f"{self.base_report_path}/{self.seed_name}_{node.generation}_{node.index}.json", "w", encoding="utf-8")
                json.dump(d, f, indent=2)
                f.close()
        current_gen_tree_nodes = current_gen_tree_nodes[:self.max_each_layer]
        return current_gen_tree_nodes

    def start(self):
        candidate_blocks = copy.deepcopy(self.seed_model.graph)
        template = copy.deepcopy(self.seed_model)
        template.graph.clear()

        root = TreeNode()
        root.new_node(self.base_output_path, 0, 1, self.seed_name)
        root.run_result = True
        root.train_result = False
        root.father = (-1, -1)
        root.save_case(template)
        root.output_shape = utils.get_input_shape_str(self.seed_name)

        last_gen = [root]
        current_gen = []

        for (i, block) in enumerate(candidate_blocks):
            current_gen += self._start_generation(last_gen, block, i+1)
            last_gen = copy.deepcopy(current_gen)
            current_gen = []


if __name__ == "__main__":
    a = Assembler("lenet")
    a.start()
