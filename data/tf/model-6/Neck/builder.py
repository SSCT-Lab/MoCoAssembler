import tensorflow as  tf

def NeckBuild(config):
    if config["model_config"]["neck"]["name"].upper() in ["FPN", "FPNLITE"]:
        from model.Neck.FPN import FPN
        return FPN
    elif config["model_config"]["neck"]["name"].upper() in ["UPUP"]:
        from model.Neck.UpUp import UPUP
        return UPUP
    else:
        raise ValueError(config["model_config"]["neck"]["name"] + " is not implemented yet or misspelled")