import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from airnetpack.blocks import Dense,MLP
from airnetpack.base import Aggregate,Filter,CFconv
from airnetpack.base import GatherNeighbors
from airnetpack.base import PositionalEmbedding
from airnetpack.base import MultiheadAttention
from airnetpack.base import Pondering,ACTWeight
from airnetpack.activations import ShiftedSoftplus,Swish
from airnetpack import Properties

class Interaction(nn.Cell):
    def __init__(self,gather_dim,fixed_neigh):
        super().__init__()

        self.fixed_neigh = fixed_neigh
        self.gather_neighbors = GatherNeighbors(gather_dim,fixed_neigh)
    
    def _set_fixed_neighbors(self,flag=True):
        self.fixed_neigh = flag
        self.gather_neighbors.fixed_neigh = flag
        return flag

class SchNetInteraction(Interaction):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_input (int): number of input atomic vector dimensions.
        dim_filter (int): number of filter dimensions.
        cfconv_module (nn.Cell): the algorthim to calcaulte continuous-filter 
            convoluations.
        cutoff_network (nn.Cell, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_input,
        num_rbf,
        dim_filter,
        activation=ShiftedSoftplus(),
        fixed_neigh=False,
        normalize_filter=False,
        axis=-2,
    ):
        super().__init__(
            gather_dim=dim_filter,
            fixed_neigh=fixed_neigh
            )
        self.atomwise_bc = Dense(n_input, dim_filter)
        self.atomwise_ac = MLP(dim_filter, n_input, [n_input,],activation=activation,use_last_activation=False)

        self.cfconv = CFconv(num_rbf,dim_filter,activation)
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def construct(self, x, e, f_ii, f_ij, c_ij, neighbors, pairwise_mask):
        """Compute convolution block.

        Args:
            x (ms.Tensor[float]): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            rbf (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (ms.Tensor[bool]): mask to filter out non-existing neighbors
                introduced via padding.

        Returns:
            ms.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        ax = self.atomwise_bc(x)
        xij = self.gather_neighbors(ax,neighbors)

        # CFconv: pass expanded interactomic distances through filter block
        y = self.cfconv(xij,f_ij,c_ij)
        # element-wise multiplication, aggregating and Dense layer
        y = self.agg(y, pairwise_mask)

        v = self.atomwise_ac(y)
        
        x_new = x + v

        return x_new, None

class AirNetInteraction(Interaction):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_input (int): number of input atomic vector dimensions.
        dim_filter (int): number of filter dimensions.
        cfconv_module (nn.Cell): the algorthim to calcaulte continuous-filter 
            convoluations.
        cutoff_network (nn.Cell, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
    """

    def __init__(
        self,
        dim_atom_embed,
        num_rbf,
        n_heads=8,
        activation=Swish(),
        max_cycles=10,
        time_embedding=0,
        use_pondering=True,
        fixed_cycles=False,
        use_filter=True,
        inside_filter=None,
        act_threshold = 0.9,
        fixed_neigh=False,
    ):
        super().__init__(
            gather_dim=dim_atom_embed,
            fixed_neigh=fixed_neigh
            )
        if dim_atom_embed % n_heads != 0:
            raise ValueError('The term "dim_atom_embed" cannot be divisible '+
                'by the term "n_heads" in AirNetIneteraction! ')

        self.n_heads=n_heads
        self.max_cycles = max_cycles
        self.dim_atom_embed = dim_atom_embed
        self.num_rbf = num_rbf
        self.time_embedding = time_embedding
                
        if fixed_cycles:
            self.flexable_cycels = False
        else:
            self.flexable_cycels = True
        
        self.use_filter = use_filter
        if self.use_filter:
            # self.filter = Filter(num_rbf,dim_atom_embed,activation)
            self.filter = Dense(num_rbf,dim_atom_embed,has_bias=True,activation=None)

        self.positional_embedding=PositionalEmbedding(dim_atom_embed)
        self.multi_head_attention=MultiheadAttention(dim_atom_embed,n_heads)

        self.act_threshold = act_threshold
        self.act_epsilon = 1.0 - act_threshold
        
        self.use_pondering = use_pondering
        self.pondering = None
        self.act_weight = None
        if self.max_cycles > 1:
            if self.use_pondering:
                self.pondering = Pondering(dim_atom_embed*3,bias_const=3)
                self.act_weight = ACTWeight(self.act_threshold)
            else:
                if self.flexable_cycels:
                    raise ValueError('The term "fixed_cycles" must be True '+
                        'when the pondering network is None in AirNetIneteraction! ')
        self.fixed_weight = Tensor(1.0 / max_cycles,ms.float32)
        
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.concat = P.Concat(-1)
        self.pack = P.Pack()
        self.reducesum = P.ReduceSum()
        self.squeeze = P.Squeeze(-1)
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.zeros = P.Zeros()

    def _ego_attention(self,x,neighbors,g_ii,g_ij,t,c_ij,mask):

        xij = self.gather_neighbors(x,neighbors)
        Q, K, V = self.positional_embedding(x,g_ii,xij,g_ij,t,c_ij)
        v = self.multi_head_attention(Q,K,V,c_ij,mask)
        return x + v

    def construct(self, x, e, f_ii, f_ij, c_ij, neighbors, mask):
        """Compute convolution block.

        Args:
            x (ms.Tensor[float]): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (ms.Tensor[bool]): mask to filter out non-existing neighbors
                introduced via padding.

        Returns:
            ms.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        
        if self.use_filter:
            g_ii = self.filter(f_ii)
            g_ij = self.filter(f_ij)
        else:
            g_ii = f_ii
            g_ij = f_ij

        # print(g_ij)
        # exit()
        
        if self.max_cycles == 1:
            t = self.time_embedding[0]
            return self._ego_attention(x,neighbors,g_ii,g_ij,t,c_ij,mask), None

        else:
            xx = x
            x0 = self.zeros_like(x)

            halting_prob = self.zeros((x.shape[0],x.shape[1]),ms.float32)
            n_updates = self.zeros((x.shape[0],x.shape[1]),ms.float32)

            broad_zeros = self.zeros_like(e)

            # cycle = self.zeros((1,),ms.int32)
            # while((halting_prob < self.act_threshold).any() and (cycle < self.max_cycles)):
            for cycle in range(self.max_cycles):
                t = self.time_embedding[cycle]
                vt = broad_zeros + t
                
                xp = self.concat((xx,e,vt))
                p = self.pondering(xp)
                # rest_prob = F.expand_dims(1.0 - halting_prob,-1)
                w, dp, dn = self.act_weight(p,halting_prob,n_updates)
                halting_prob = halting_prob + dp
                n_updates = n_updates + dn

                xx = self._ego_attention(xx,neighbors,g_ii,g_ij,t,c_ij,mask)

                cycle = cycle + 1

                # tensor_cycle = self.ones_like(w) * cycle
                # tensor_max = self.ones_like(w) * self.max_cycles
                # w = F.select(tensor_cycle<tensor_max,w,rest_prob)

                x0 = xx * w + x0 * (1.0 - w)

            return x0, n_updates
