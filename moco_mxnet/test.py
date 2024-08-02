from Tools.Parser import GetSeed
lenet = GetSeed("LeNet")
lenet.AssembleFile("./Output", "tlenet")
lenet = GetSeed("AlexNet")
lenet.AssembleFile("./Output", "talexnet")
lenet = GetSeed("PointNet")
lenet.AssembleFile("./Output", "tpointnet")
lenet = GetSeed("SqueezeNet")
lenet.AssembleFile("./Output", "tsqueezenet")
