# BUG LIST for TensorFlow

| Bug API                | Bug Type | File | Error Message                                                |
| ---------------------- | -------- | ---- | ------------------------------------------------------------ |
| ActivityRegularization | BonBug   |      | The l1 and l2 parameters of the ActivityRegularization function are described in this document as floating point numbers and their values should be positive. |
| GRU                    | BonBug   |      | The document describes the dropout parameters of the GRU/LSTM/SimpleRNN function and the recurrent_dropout parameters are floating-point numbers with values ranging from 0 to 1, but the program can still run normally if the number is not between 0 and 1. |
| LSTM                   | BonBug   |      | The document describes the dropout parameters of the GRU/LSTM/SimpleRNN function and the recurrent_dropout parameters are floating-point numbers with values ranging from 0 to 1, but the program can still run normally if the number is not between 0 and 1. |
| SimpleRNN              | BonBug   |      | The document describes the dropout parameters of the GRU/LSTM/SimpleRNN function and the recurrent_dropout parameters are floating-point numbers with values ranging from 0 to 1, but the program can still run normally if the number is not between 0 and 1. |
| LeakyReLU              | BonBug   |      | The document describes the LeakyReLU function as having an alpha parameter whose type is a floating point number and the value is equal to or greater than 0. But we found that the value of this parameter is less than zero can also work. |
| Cropping2D             | BonBug   |      | When the value type of Cropping in Cropping2D is int, tf doesn't check whether Cropping is positive or negative. |
| Cropping3D             | BonBug   |      | When the value type of Cropping in Cropping3D is int, tf doesn't check whether Cropping is positive or negative. |
| ZeroPadding1D          | ICBug    |      | ValueError: The `padding` argument must be a tuple of 2 integers. Received: {'padding': 1}. |
| LayerNormalization     | ICBug    |      | Epsilon is documented as a small floating-point number.In our experiments, we found that epsilon works with larger floating-point numbers. |
| LayerNormalization     | ImpBug   |      | Epsilon is used to avoid division by zero errors, but when LayerNormalization takes a zero vector as input. |
| -                      | PerBug   |      | We have many similar model building and training code files, and train multiple models consecutively through one process. But in this process, we used the memory blank and the restriction function, and neither worked. |

