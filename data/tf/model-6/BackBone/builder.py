import tensorflow as tf

def BackBoneBuild(config):
    if config["model_config"]["backbone"]["name"].upper() in ["MOBILEDET"]:
        from model.BackBone.MobileDet import MobileDetCPU
        backbone = MobileDetCPU

    elif config["model_config"]["backbone"]["name"].upper() in ["MV3", "MOBILENETV3", "MOBILENET3"]:
        if config["model_config"]["backbone"]["modelSize"].upper() == "SMALL":
            from model.BackBone.MobilenetV3 import MobileNetV3Small
            backbone = MobileNetV3Small
        else:
            from model.BackBone.MobilenetV3 import MobileNetV3Large
            backbone = MobileNetV3Large
    elif config["model_config"]["backbone"]["name"].upper() in ["EFFICIENTNET"]:
        backbone = model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, include_preprocessing=False, input_shape=(320, 320, 3))

    else:
        raise ValueError("Not implemented yet or misspelled")

    return backbone
    