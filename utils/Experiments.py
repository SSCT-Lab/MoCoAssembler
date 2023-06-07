class Experiments:

    def __init__(self):
        self.simple_model = ["lenet",
                             "alexnet",
                             "vgg16",
                             "vgg19",
                             "mobilenet"
                             ]

        self.rnn_model = ["lstm",
                          "bilstm",
                          "gru"]

        self.complex_model = ["googlenet",
                              "resnet18",
                              "resnet50",
                              "squeezenet",
                              "xception",
                              "densenet",
                              "inceptionv3"
                              ]

    def trainOne(self, model, mutate_times): pass
