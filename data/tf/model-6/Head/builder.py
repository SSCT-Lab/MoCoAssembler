import tensorflow as  tf

def HeadBuild(config):
    if config["model_config"]["head"]["name"].upper() in ["CENTERNET"]:
        from model.Head.CenterNet import CenterNet
        return CenterNet
    elif config["model_config"]["head"]["name"].upper() in ["TTFNET"]:
        from model.Head.TTFNet import TTFNet
        return TTFNet
    else:
        raise ValueError(config["model_config"]["head"]["name"] + " is not implemented yet or misspelled")