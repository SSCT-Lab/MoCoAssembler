"""
    init函数变异过程中需要用到的api：
        activation_list: 共29项，最常见的是ReLU，Softmax
        batchnorm_list: 共7项，最常见的是BatchNorm2d
        conv_list: 共12项，最常见的是Conv2d
        dropout_list: 共6项，最常见的是Dropout
        linear_list: 共4项，最常见的是Linear
        pool_list: 共19项，最常见的是MaxPool2d，AdaptiveAvgPool2d
        rnn_list: 共8项，最常见的是LSTMCell
        sparse_list: 共2项，最常见的是Embedding

    两个函数：分别对应mutate中的变异API名称和参数
"""

# activation
activation_list = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh', 'SiLU', 'Mish',
                   'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU', 'LogSigmoid',
                   'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink', 'Softmin',
                   'Softmax', 'Softmax2d', 'LogSoftmax']

# batchnorm
batchnorm_list = ['BatchNorm1d', 'LazyBatchNorm1d', 'BatchNorm2d', 'LazyBatchNorm2d', 'BatchNorm3d', 'LazyBatchNorm3d',
                  'SyncBatchNorm']

# conv
conv_list = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'LazyConv1d',
             'LazyConv2d', 'LazyConv3d', 'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d']

# dropout
dropout_list = ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout']

# linear
linear_list = ['Bilinear', 'Identity', 'LazyLinear', 'Linear']

# pool
pool_list = ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'AvgPool1d',
             'AvgPool2d', 'AvgPool3d', 'FractionalMaxPool2d', 'FractionalMaxPool3d', 'LPPool1d', 'LPPool2d',
             'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
             'AdaptiveAvgPool3d']

# rnn
rnn_list = ['RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell', 'GRUCell']

# sparse
sparse_list = ['Embedding', 'EmbeddingBag']


# TODO:更改api
def change_api(line: str) -> str:
    return "# api\n"

# TODO:更改参数值
def change_params(line: str) -> str:
    return "# params\n"
