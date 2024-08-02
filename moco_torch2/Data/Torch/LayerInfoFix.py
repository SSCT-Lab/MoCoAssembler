import copy
import os
import json


old_file_list = os.listdir("./torch_layer_info")
api_file_list = os.listdir("./torch_layer_info_new")


def c(d):
    dd = d["params"]
    defaultEntranceFlag = False
    for key in dd.keys():
        pD = dd[key]
        if "dtype" not in pD.keys() or \
                "range" not in pD.keys() or \
                "structure" not in pD.keys() or \
                "shape" not in pD.keys() or \
                "default" not in pD.keys():
            print(f" param {key} has some problem: lack key in dict.")
            break

        # check dtype
        typeList = pD["dtype"]
        if not isinstance(typeList, list):
            print(f" param {key} has some problem: dtype is not a list.")
            break
        pFlag = False
        for tp in typeList:
            if tp not in ["int", "float", "string", "boolean"]:
                pFlag = True
                break
        if pFlag:
            print(f" param {key} has some problem: wrong type in dtype list.")
            break

        # check range
        rangeList = pD["range"]
        if not isinstance(rangeList, list):
            print(f" param {key} has some problem: range is not a list.")
            break
        if len(rangeList) != len(typeList):
            print(f" param {key} has some problem: lengths of range and dtype don't equal.")
            break
        pFlag = False
        for rg in rangeList:
            if not isinstance(rg, list) and not (rg is None):
                pFlag = True
                break
        if pFlag:
            print(f" param {key} has some problem: wrong type in range list.")
            break

        # check structure
        structureList = pD["structure"]
        if not isinstance(structureList, list):
            print(f" param {key} has some problem: structure is not a list.")
            break
        pFlag = False
        for tp in structureList:
            if tp not in ["scalar", "tuple", "list"]:
                pFlag = True
                break
        if pFlag:
            print(f" param {key} has some problem: wrong type in structure list.")
            break

        # check shape
        shapeList = pD["shape"]
        if not isinstance(shapeList, list):
            print(f" param {key} has some problem: shape is not a list.")
            break
        if len(shapeList) != len(structureList):
            print(f" param {key} has some problem: lengths of structure and shape don't equal.")
            break
        pFlag = False
        for sp in shapeList:
            if not isinstance(sp, int):
                pFlag = True
                break
        if pFlag:
            print(f" param {key} has some problem: wrong type in shape list.")
            break

        # check default
        defaultValue = pD["default"]
        if not isinstance(defaultValue, int) and not isinstance(defaultValue, float) and not isinstance(
                defaultValue, str) and defaultValue is not None:
            print(f" param {key} has some problem: default has a wrong type.")
            break
        if defaultValue is None:
            if defaultEntranceFlag:
                print(f" param {key} has some problem: required param locates after optional param.")
                break
        else:
            defaultEntranceFlag = True


def f(file_name: str):
    if file_name.replace(".", "_").replace("_json", ".json") in old_file_list:
        return "no"
    f = open(f"./torch_layer_info_new/{file_name}", "r", encoding="utf-8")
    info_d = json.load(f)
    f.close()
    template_for_a_param = {
        "dtype": [],
        "range": [],
        "structure": [],
        "shape": [],
        "default": None
    }
    params_d = info_d["params"]
    params_d_new = {}
    for param in params_d.keys():
        if param == "input":
            continue
        param_d = params_d[param]
        param_d_new = copy.deepcopy(template_for_a_param)

        # rule dtype & range
        if "int" in param_d["dtype"]:
            param_d_new["dtype"].append("int")
            param_d_new["range"].append(None)
        elif "float" in param_d["dtype"]:
            param_d_new["dtype"].append("float")
            param_d_new["range"].append(None)
        elif "string" in param_d["dtype"]:
            param_d_new["dtype"].append("float")
            param_d_new["range"].append(["str1TBD", "str2TBD"])
        elif "boolean" in param_d["dtype"]:
            param_d_new["dtype"].append("boolean")
            param_d_new["range"].append(None)

        # rule structure & shape
        if "scalar" in param_d["structure"]:
            param_d_new["structure"].append("scalar")
            param_d_new["shape"].append(1)
        if "tuple" in param_d["structure"]:
            param_d_new["structure"].append("tuple")
            flag = False
            for dim in range(2, 6):
                if str(dim) in param_d["shape"]:
                    param_d_new["shape"].append(dim)
                    flag = True
                    break
            if not flag:
                param_d_new["shape"].append(1)
                if "1d" not in file_name:
                    print(f"HHHHHHHHHH {file_name} may has some shape problems in param {param} (tuple), check it.")
        if "list" in param_d["structure"]:
            param_d_new["structure"].append("structure")
            flag = False
            for dim in range(2, 6):
                if str(dim) in param_d["shape"]:
                    param_d_new["shape"].append(dim)
                    flag = True
                    break
            if not flag:
                param_d_new["shape"].append(1)
                if "1d" not in file_name:
                    print(f"HHHHHHHHHH {file_name} may has some shape problems in param {param} (list), check it.")

        params_d_new[param] = param_d_new
    return {
        "params": params_d_new,
        "constraints": info_d["constraints"]
    }


def read_and_save(fileName):
    d = f(fileName)
    if not isinstance(d, dict):
        return
    else:
        d_new = d
        print(fileName + " : ==================")
        c(d_new)
        ff = open(f"./torch_layer_info_new_new/{fileName}", "w", encoding="utf-8")
        json.dump(d_new, ff, indent=2)
        ff.close()
        return


if __name__ == "__main__":
    read_and_save("torch.nn.Conv2d.json")
    # for FileName in api_file_list:
    #     try:
    #         read_and_save(FileName)
    #     except Exception:
    #         print(f"********************{FileName} bad json************************")
