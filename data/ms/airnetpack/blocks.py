import mindspore
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.layer.activation import get_activation

# ~ from schnetpack import Properties
from airnetpack.activations import Swish
from airnetpack import Properties

__all__ = ["Dense","MLP", "ElementalGate"]

# class Dense(nn.Dense):
#         def __init__(self,
#         in_channels,
#         out_channels,
#         weight_init='xavier_uniform',
#         bias_init='zero',
#         has_bias=True,
#         activation=None,
#     ):
#         super().__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             weight_init=weight_init,
#             bias_init=bias_init,
#             has_bias=has_bias,
#             activation=activation
#         )

class Dense(nn.Cell):
    def __init__(self,
        in_channels,
        out_channels,
        weight_init='xavier_uniform',
        bias_init='zero',
        has_bias=True,
        activation=None,
        do_reshape=True,
    ):
        super().__init__()
        self.do_reshape = do_reshape
        self.dense = nn.Dense(
            in_channels=in_channels,
            out_channels=out_channels,
            weight_init=weight_init,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=activation
        )

    def construct(self,x):
        shape = x.shape
        if self.do_reshape:
            x_shape = (-1,shape[-1])
            x = F.reshape(x,x_shape)
        
        y = self.dense(x)

        if self.do_reshape:
            y_shape = shape[:-1] + (-1,)
            y = F.reshape(y,y_shape)

        return y

class MLP(nn.Cell):
    """Multiple layer fully connected perceptron neural network.

    Args:
        n_in (int): number of input dimensions.
        n_out (int): number of output dimensions.
        layer_dims (list of int or int): number hidden layer dimensions.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.

    """

    def __init__(
        self,
        n_in,
        n_out,
        layer_dims=None,
        activation=None,
        weight_init='xavier_uniform',
        bias_init='zero',
        use_last_activation=False,
        do_reshape=True,
        ):
        super().__init__()

        self.do_reshape = do_reshape

        # get list of number of dimensions in input, hidden & output layers
        if layer_dims is None or len(layer_dims)==0:
            self.mlp = nn.Dense(n_in, n_out, activation=activation)
        else:
            # assign a Dense layer (with activation function) to each hidden layer
            nets=[]
            indim=n_in
            for ldim in layer_dims:
                # nets.append(Dense(indim, ldim,activation=activation))
                nets.append(
                    nn.Dense(
                    in_channels=indim,
                    out_channels=ldim,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    has_bias=True,
                    activation=activation
                    )
                )
                indim=ldim

            # assign a Dense layer to the output layer
            if use_last_activation and activation is not None:
                nets.append(
                    nn.Dense(
                    in_channels=indim,
                    out_channels=n_out,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    has_bias=True,
                    activation=activation)
                )
            else:
                nets.append(
                    nn.Dense(
                    in_channels=indim,
                    out_channels=n_out,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    has_bias=True,
                    activation=None)
                )
            # put all layers together to make the network
            self.mlp = nn.SequentialCell(nets)

    def construct(self, x):
        """Compute neural network output.

        Args:
            inputs (torch.Tensor): network input.

        Returns:
            torch.Tensor: network output.

        """
        shape = x.shape
        if self.do_reshape:
            x_shape = (-1,shape[-1])
            x = F.reshape(x,x_shape)
        
        y = self.mlp(x)
        
        if self.do_reshape:
            y_shape = shape[:-1] + (-1,)
            y = F.reshape(y,y_shape)

        return y

class ElementalGate(nn.Cell):
    """
    Produces a Nbatch x Natoms x Nelem mask depending on the nuclear charges passed as an argument.
    If onehot is set, mask is one-hot mask, else a random embedding is used.
    If the trainable flag is set to true, the gate values can be adapted during training.

    Args:
        elements (set of int): Set of atomic number present in the data
        onehot (bool): Use one hit encoding for elemental gate. If set to False, random embedding is used instead.
        trainable (bool): If set to true, gate can be learned during training (default False)
    """

    def __init__(self, elements, onehot=True, trainable=False):
        super().__init__()
        self.trainable = trainable

        # Get the number of elements, as well as the highest nuclear charge to use in the embedding vector
        self.nelems = len(elements)
        maxelem = int(max(elements) + 1)

        self.gate = nn.Embedding(maxelem, self.nelems, onehot)

        # Set trainable flag
        if not trainable:
            self.gate.embedding_table.requires_grad = False

    def construct(self, atomic_numbers):
        """
        Args:0
            atomic_numbers (torch.Tensor): Tensor containing atomic numbers of each atom.

        Returns:
            torch.Tensor: One-hot vector which is one at the position of the element and zero otherwise.

        """
        return self.gate(atomic_numbers)
