import json
import pickle
import sys
import time
import os
import importlib.util
import traceback

from DS.Model import Model
from Tools.Mutator import Mutator


class Boundary:
    def __init__(self, seed, mutate_file="/Users/wuduo/Documents/BioWork/MoCo/MOCO-3/MoCoAssembler/moco_tf2/output", output_path="/Users/wuduo/Documents/BioWork/MoCo/MOCO-3/MoCoAssembler/moco_tf2/boundary"):
        self.seed_name = seed
        self.mutate_file = mutate_file

        self.base_output_path = f"{output_path}/{self.seed_name}/tree"
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(self.base_output_path, exist_ok=True)

        sys.path.append(self.base_output_path)

        self.base_report_path = f"{output_path}/{self.seed_name}/report"
        os.makedirs(self.base_report_path, exist_ok=True)

        self.mutator = Mutator()

    def start(self):
        mutate_dir = os.path.join(self.mutate_file, self.seed_name)
        if not os.path.exists(mutate_dir):
            print("{} file does not exist.".format(mutate_dir))
            return

        for root, dirs, files in os.walk(mutate_dir):
            for file in files:
                mutate_file = os.path.join(root, file)
                if file.endswith("go.py"):
                    pkl_path = mutate_file.replace(mutate_file.split('/')[-1], "model.pkl")

                    with open(pkl_path, 'rb') as f:
                        model = pickle.load(f)
                    block = model.graph[-1]
                    all_block = model.graph[ : -1]

                    res, res_tag = self.mutator.boundary_mutate(block)
                    for i in range(len(res)):
                        all_block.append(res[i])
                        new_model = Model(all_block, model.model_inputs, model.model_outputs)
                        new_model.model_name = model.model_name
                        new_file = file.replace("go", str(i)).split('.')[0]
                        new_model.assemble_file(self.base_output_path, new_file, True, False)
                        run_time, run_error_info = self.run(os.path.join(self.base_output_path, new_file))
                        run_result = (run_time >= 0.0)
                        result = {
                            "go result": run_result,
                            "go time": run_time,
                            "go error info": run_error_info,
                            "tag": res_tag[i]
                        }

                        with open(os.path.join(self.base_report_path, f"{new_file}.json"), "w") as f:
                            json.dump(result, f, indent=2)
                        exit()

    def run(self, case_path):
        start = time.time()
        module_name = case_path.split('/')[-1]
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{case_path}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self.output_shape = module.go().shape
            error_message = ""
            end = time.time()
            run_time = end - start
        except Exception as e:
            run_time = -1
            error_message = traceback.format_exc()

        return run_time, error_message
