pool_list = ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'AvgPool1d',
             'AvgPool2d', 'AvgPool3d', 'FractionalMaxPool2d', 'FractionalMaxPool3d', 'LPPool1d', 'LPPool2d',
             'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
             'AdaptiveAvgPool3d']

activation_list = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh', 'SiLU', 'Mish',
                   'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU', 'LogSigmoid',
                   'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink', 'Softmin',
                   'Softmax', 'Softmax2d', 'LogSoftmax']

dropout_list = ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout']

conv_list = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'LazyConv1d',
             'LazyConv2d', 'LazyConv3d', 'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d']

linear_list = ['Bilinear', 'Identity', 'LazyLinear', 'Linear']

sparse_list = ['Embedding', 'EmbeddingBag']

rnn_list = ['RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell', 'GRUCell']

batchnorm_list = ['BatchNorm1d', 'LazyBatchNorm1d', 'BatchNorm2d', 'LazyBatchNorm2d', 'BatchNorm3d', 'LazyBatchNorm3d',
                  'SyncBatchNorm']


def changeAPI(line: str) -> str:
    return line
