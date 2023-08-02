| DL Library |                                          issue                                          |                          url                          |  status   |
|:----------:|:---------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:---------:|
| TensorFlow |        Crooping2D/3D does not have exception handling for the Crooping parameter        | https://github.com/tensorflow/tensorflow/issues/61255 | confirmed |
| TensorFlow |                   Documentation Bug about API ActivityRegularization                    | https://github.com/tensorflow/tensorflow/issues/61254 |   fixed   |
| TensorFlow |                      Functions that limit video memory do not work                      | https://github.com/tensorflow/tensorflow/issues/61168 | confirmed |
| TensorFlow |                               the description of padding                                | https://github.com/tensorflow/tensorflow/issues/60839 |   fixed   |
| TensorFlow |          The return value when LayerNormalization takes zero vectors as input           | https://github.com/tensorflow/tensorflow/issues/61287 | confirmed |
| TensorFlow |                   The type of epsilon parameter of LayerNormalization                   | https://github.com/tensorflow/tensorflow/issues/61287 | confirmed |
| TensorFlow |                     The value range of parameters of GRU functions                      | https://github.com/tensorflow/tensorflow/issues/61256 | confirmed |
| TensorFlow |                The legal value range of the alpha parameter in LeakyReLU                | https://github.com/tensorflow/tensorflow/issues/61257 |   fixed   |
| TensorFlow |                          The value range of parameters of LSTM                          | https://github.com/tensorflow/tensorflow/issues/61256 | confirmed |
| TensorFlow |                   Could not interpret serialized activation function                    | https://github.com/tensorflow/tensorflow/issues/60840 | confirmed |
| TensorFlow |                       The value range of parameters of SimpleRNN                        | https://github.com/tensorflow/tensorflow/issues/61256 | confirmed |
|  PyTorch   |            torch.nn.Conv2d's padding mode circular cannot accept 3-dim input            |   https://github.com/pytorch/pytorch/issues/104860    |   fixed   |
|  PyTorch   |              torch.nn.MultiheadAttention lacks parameter validation check               |   https://github.com/pytorch/pytorch/issues/105630    |   fixed   |
|  PyTorch   |                           "padding" dimensions of Pad Layers                            |   https://github.com/pytorch/pytorch/issues/105627    |   fixed   |
|  PyTorch   |                             Input dimensions of Pad Layers                              |   https://github.com/pytorch/pytorch/issues/105627    |   fixed   |
|  PyTorch   |              Incomplete Documentation for torch.nn.FractionalMaxPool2d API              |   https://github.com/pytorch/pytorch/issues/104861    |   fixed   |
|  PyTorch   |              The document does not emphasize hidden range in nn.MaxPool2d               |   https://github.com/pytorch/pytorch/issues/103423    | confirmed |
|  PyTorch   |              The document does not emphasize hidden range in nn.Embedding               |   https://github.com/pytorch/pytorch/issues/103424    | confirmed |
|  PyTorch   |                  F.pad will accept 0 and negative values as parameter                   |   https://github.com/pytorch/pytorch/issues/105629    | confirmed |
|  PyTorch   |                      Incomplete Documentation for torch.nn.RNNBase                      |   https://github.com/pytorch/pytorch/issues/105628    | fixed     |
|  PyTorch   |              torch.nn.MultiheadAttention lacks parameter validation check               |   https://github.com/pytorch/pytorch/issues/105630    | confirmed |
|  PyTorch   |            torch.nn.TransformerDecoderLayer lacks parameter validation check            |   https://github.com/pytorch/pytorch/issues/105632    | confirmed |
|   Jittor   |                            jittor.nn.Mish is unable to work                             |      https://github.com/Jittor/jittor/issues/447      |   fixed   |
|   Jittor   |      jittor.nn.Flatten accepts the input tensor while cannot cover all dimensions       |      https://github.com/Jittor/jittor/issues/448      | confirmed |
|   Jittor   |              jittor.nn.Upsample is unable to work with default parameters               |      https://github.com/Jittor/jittor/issues/450      | confirmed |
|   Jittor   |       jittor.nn.MaxPool2d's special value causes compilation failure of operators       |      https://github.com/Jittor/jittor/issues/451      | confirmed |
|   Jittor   |             jittor.nn.AdaptiveMaxPool3d throw an error for compliant inputs             |      https://github.com/Jittor/jittor/issues/452      | confirmed |
|   Jittor   |                jittor.nn.AdaptiveMaxPool3d has errors in its source code                |      https://github.com/Jittor/jittor/issues/453      | confirmed |
|   Jittor   |     jittor.nn.MaxPool2d accepts illegal parameters and causes illegal computations      |      https://github.com/Jittor/jittor/issues/456      | confirmed |
|   Jittor   |                        jittor.nn.Pool accepts illegal parameters                        |      https://github.com/Jittor/jittor/issues/457      | confirmed |
|   Jittor   |                    jittor.nn.PixelShuffle accepts illegal parameters                    |      https://github.com/Jittor/jittor/issues/458      | confirmed |
|   Jittor   |                    jittor.cat generates incorrect exception messages                    |      https://github.com/Jittor/jittor/issues/459      | confirmed |
|   Jittor   |           jittor.nn.AdaptiveMaxPool3d generates incorrect exception messages            |      https://github.com/Jittor/jittor/issues/460      | confirmed |
|   Jittor   |                   jittor.nn.Resize accepts illegal parameters "size"                    |      https://github.com/Jittor/jittor/issues/461      | confirmed |
|   Jittor   |            jittor.nn.Resize is unable to calculate tensors with 0 batch-size            |      https://github.com/Jittor/jittor/issues/462      | confirmed |
|   Jittor   |              jittor.nn.AvgPool2d has illegal operations during calculating              |      https://github.com/Jittor/jittor/issues/463      | confirmed |
|   Jittor   |                  jittor.nn.ReflectionPad2d accepts illegal parameters                   |      https://github.com/Jittor/jittor/issues/464      | confirmed |
|   Jittor   |     jittor.nn.ZeroPad2d accepts illegal parameters and causes the operator to crash     |      https://github.com/Jittor/jittor/issues/465      | confirmed |
|   Jittor   | jittor.nn.ReplicationPad2d accepts illegal parameters and causes the operator to crash  |      https://github.com/Jittor/jittor/issues/466      | confirmed |
|   Jittor   |   jittor.nn.ConstantPad2d accepts illegal parameters and causes the operator to crash   |      https://github.com/Jittor/jittor/issues/467      | confirmed |
|   Jittor   |             jittor.nn.LSTM is unable to respond correctly to illegal inputs             |      https://github.com/Jittor/jittor/issues/468      | confirmed |
|   Jittor   |            jittor.nn.AdaptiveMaxPool2d has incorrect handling of dimensions             |      https://github.com/Jittor/jittor/issues/469      | confirmed |
|   Jittor   |      jittor.nn.AdaptiveMaxPool2d lacks input dimension checks in specific branches      |      https://github.com/Jittor/jittor/issues/470      | confirmed |
|   Jittor   |      jittor.nn.Conv2d accepts illegal parameters and causes the operator to crash       |      https://github.com/Jittor/jittor/issues/471      | confirmed |
|   Jittor   |            jittor.nn.Conv2d's parameter "kernel_size" accepts illegal values            |      https://github.com/Jittor/jittor/issues/472      | confirmed |
|   Jittor   |              jittor.nn.Conv2d's parameter "stride" accepts illegal values               |      https://github.com/Jittor/jittor/issues/473      | confirmed |
|   Jittor   |             jittor.nn.Conv2d's parameter "dilation" accepts illegal values              |      https://github.com/Jittor/jittor/issues/474      | confirmed |
|   Jittor   |              jittor.nn.Conv2d's parameter "groups" accepts illegal values               |      https://github.com/Jittor/jittor/issues/475      | confirmed |
|   Jittor   |            jittor.nn.Conv2d's parameter "in_channels" accepts illegal values            |      https://github.com/Jittor/jittor/issues/476      | confirmed |
|   Jittor   |           jittor.nn.Conv2d's parameter "out_channels" accepts illegal values            |      https://github.com/Jittor/jittor/issues/477      | confirmed |
|   Jittor   |          jittor.nn.ConvTranspose's parameter "padding" accepts illegal values           |      https://github.com/Jittor/jittor/issues/478      | confirmed |
|   Jittor   | jittor.nn.ConvTranspose has illegal operations while setting "padding" to a large value |      https://github.com/Jittor/jittor/issues/479      | confirmed |
|   Jittor   |          jittor.nn.MaxPool2d's parameter "kernel_size" accepts illegal values           |      https://github.com/Jittor/jittor/issues/480      | confirmed |
|   Jittor   |             jittor.nn.MaxPool2d's parameter "stride" accepts illegal values             |      https://github.com/Jittor/jittor/issues/481      | confirmed |
|   Jittor   |             jittor.nn.Pool's parameter "kernel_size" accepts illegal values             |      https://github.com/Jittor/jittor/issues/482      | confirmed |
|   Jittor   |               jittor.nn.Pool's parameter "stride" accepts illegal values                |      https://github.com/Jittor/jittor/issues/483      | confirmed |
|   Jittor   |          jittor.nn.AdaptiveMaxPool3d's documentation is missing on the website          |      https://github.com/Jittor/jittor/issues/484      | confirmed |
