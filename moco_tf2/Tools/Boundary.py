import copy
import pickle
import time
import os

from Tools.Assembler import TreeNode
from Tools.Mutator import Mutator
from Tools.Parser import get_seed


class Boundary:
    def __init__(self, seed, experiment_name=None, output_path="./output", fuzzing_path=None):
        if experiment_name is None:
            self.experiment_name = seed + str(time.time()) + "_boundary"
        else:
            self.experiment_name = experiment_name
        self.base_output_path = f"{output_path}/{experiment_name}/tree"
        self.base_report_path = f"{output_path}/{experiment_name}/report"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(self.base_output_path, exist_ok=True)
        os.makedirs(self.base_report_path, exist_ok=True)

        self.seed_name = seed
        self.seed_model = get_seed(seed)
        self.fuzzing_path = f"{output_path}/{fuzzing_path}"

        self.mutator = Mutator()

    def _start_generation(self, template, block, gen):
        father = TreeNode()
        father.new_node(self.base_output_path, gen, 0, self.seed_name)
        father.save_case(template)
        father.run()
        father.spawn_boundary(block)

    def start(self):
        candidate_blocks = copy.deepcopy(self.seed_model.graph)
        template = copy.deepcopy(self.seed_model)
        template.graph.clear()

        for (i, block) in enumerate(candidate_blocks):
            if block.is_child_model:
                template.graph.append(block)
                continue
            template2 = copy.deepcopy(template)
            template2.model_outputs = block.output_symbols
            self._start_generation(template2, block, i + 1)
            template.model_outputs = block.output_symbols
            template.graph.append(block)

    def start2(self):
        if not os.path.exists(self.fuzzing_path):
            return
        tree_path = os.path.join(self.fuzzing_path, "tree")
        layers = os.listdir(tree_path)
        for gen in layers:
            api_set = set()
            gen_path = os.path.join(tree_path, gen)
            print(gen_path)
            for root, dirs, files in os.walk(gen_path):
                for file in files:
                    if file.endswith(".pkl"):
                        print(os.path.join(root, file))
                        with open(os.path.join(root, file), "rb") as f:
                            model = pickle.load(f)
                        block = model.graph[-1]
                        if block.is_child_model or block.api_name in api_set:
                            continue

                        father = TreeNode()
                        father.new_node(self.base_report_path, gen, 0, self.seed_name)

            break
