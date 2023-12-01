import copy
import os
import yaml
from _database import database
import os.path as p
from Concrete import translator_factory


library = database._Database__library_list[0]
translator = translator_factory(library)
translate = translator.translate
no_mutate_pool = ['in_channels', 'out_channels', 'in_features', 'out_features', 'input_size', 'output_size',
                  'num_features']


def generateBoundaryCases(case_path: str, abstract_model: dict) -> int:
    model_name = list(abstract_model.keys())[0]
    boundary_case_path = p.join(case_path, "boundary_cases")
    if not p.exists(boundary_case_path):
        os.makedirs(boundary_case_path)
    if len(abstract_model[model_name]) <= 1:
        # don't bound this
        return 0
    layer_to_bound = abstract_model[model_name][-1]
    if ("layer" not in list(layer_to_bound.keys())) or \
            (not database.is_abstract_api_name_valid(layer_to_bound["layer"])):
        # don't bound this
        return 0
    else:
        abstract_layer_name = layer_to_bound["layer"]
        abstract_layer_info = database.get_abstract_layer_info(abstract_layer_name)
        constraints = abstract_layer_info["constraints"]
        para_list = list(layer_to_bound["params"].keys())
        _ = copy.deepcopy(para_list)
        new_layer_list = []
        for para_name in _:
            if para_name in no_mutate_pool:
                para_list.remove(para_name)
        for para in para_list:
            if 'dtype' not in constraints[para].keys():
                continue
            elif ('int' not in constraints[para]['dtype']) and ('tuple' not in constraints[para]['dtype']):
                continue
            else:
                if "range" in constraints.keys() and isinstance(constraints["range"], list) \
                        and len(constraints["range"] == 2):
                    _min, _max = constraints["range"][0], constraints["range"][1]
                elif para == 'dilation':
                    _min, _max = 1, 6
                elif para == 'padding':
                    _min, _max = 0, 8
                elif para == 'groups':
                    _min, _max = 1, 3
                elif para == 'kernel_size':
                    _min, _max = 1, 8
                elif para == 'stride':
                    _min, _max = 1, 4
                elif 'channels' in para or 'features' in para or 'size' in para:
                    _min, _max = 1, 512
                else:
                    _min, _max = 1, 8
                min_value = _min
                max_value = _max
                int_pool = [(min_value, 1), (min_value - 1, 0), (min_value + 1, 1),
                            (max_value, 1), (max_value - 1, 1), (max_value + 1, 2)]
                if ((library == "torch" or "tensorflow") and "integer" in constraints[para]["structure"]) or \
                        (library == "jittor" and "int" in constraints[para]["dtype"]):
                    for value in int_pool:
                        new_layer = copy.deepcopy(layer_to_bound)
                        new_layer["params"][para] = value[0]
                        new_layer_list.append((new_layer, value[1]))
                if ((library == "torch" or "tensorflow") and "tuple" in constraints[para]["structure"]) or \
                        (library == "jittor" and "tuple" in constraints[para]["dtype"]):
                    tuple_pool = []
                    for i in int_pool:
                        for j in int_pool:
                            t = (i[0], j[0])
                            if i[1] == 0 or j[1] == 0:
                                exp = 0
                            elif i[1] == 2 or j[1] == 2:
                                exp = 2
                            else:
                                exp = 1
                            tuple_pool.append((t, exp))
                    for value in tuple_pool:
                        new_layer = copy.deepcopy(layer_to_bound)
                        new_layer["params"][para] = value[0]
                        new_layer_list.append((new_layer, value[1]))
        with open(p.join(boundary_case_path, "boundary_report.txt"), "w", encoding="utf-8") as f:
            f.write("num,layer,expect result,true result\n")
        f_report = open(p.join(boundary_case_path, "boundary_report.txt"), "a", encoding="utf-8")
        for i in range(len(new_layer_list)):
            new_layer, expect_result = new_layer_list[i]
            new_model = copy.deepcopy(abstract_model)
            new_model[model_name][-1] = new_layer
            model_code = translator.translate(new_model)
            with open(p.join(boundary_case_path, str(i+1) + ".py"), "w", encoding="utf-8") as f_py:
                f_py.write(model_code)
            if expect_result == 0:
                expect_result = "FAIL"
            elif expect_result == 1:
                expect_result = "SUCCESS"
            else:
                expect_result = "DEPEND"
            report = str(i+1) + "," + str(new_layer) + "," + expect_result + "," + "\n"
            f_report.write(report)
        f_report.close()
        return len(new_layer_list)


def generateBoundary(experiment_id: str):
    if experiment_id.startswith("experiment"):
        exp_id = experiment_id
    else:
        exp_id = "experiment" + experiment_id
    experiment_path = p.join("..", "result", exp_id)
    case_list = os.listdir(experiment_path)
    if len(case_list) <= 2:
        print("NO BOUNDARY")
        return
    model_name = "default"
    for case in case_list:
        if not case.startswith("main_report"):
            model_name = case.split("-", 1)[0]
            break
    if model_name == "default":
        print("NO BOUNDARY")
        return
    case_list.remove("main_report.txt")
    for case in case_list:
        current_case_path = p.join(experiment_path, case)
        if "MoCoNA.py" not in os.listdir(current_case_path):
            continue
        else:
            with open(p.join(current_case_path, case + "_abstract.yaml"), "r", encoding="utf-8") as f:
                model = yaml.full_load(f)
            generateBoundaryCases(current_case_path, model)
            print("generate boundary cases for " + case)
    return


if __name__ == "__main__":
    case_path = "D:/PythonProjects/MoCo_Delta/MoCo_F2/result/experiment1699202204/LeNet-3-4"
    abstract_model_path = p.join(case_path, "LeNet-3-4_abstract.yaml")
    with open(abstract_model_path, "r", encoding="utf-8") as f:
        abstract_model = yaml.full_load(f)
    num = generateBoundaryCases(case_path, abstract_model)
