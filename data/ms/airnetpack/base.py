import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import Constant

from airnetpack.blocks import MLP,Dense
from airnetpack.activations import Swish

__all__ = ["AtomwiseError","AtomwiseReadout","MultipleChannelRepresentation", "CFconv", "ScaleShift", "Standardize", "Aggregate"]
            

class AtomwiseError(Exception):
    pass

class AtomwiseReadout(nn.Cell):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        n_neurons=[32,],
        activation=Swish(),
        outnet=None,
    ):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out

        # build output network
        if outnet is None:
            self.out_net = MLP(n_in, n_out, n_neurons, activation=activation)
        else:
            self.out_net = outnet

    def construct(self, inputs, atom_mask=None):
        r"""
        predicts atomwise property
        """
        
        return self.out_net(inputs)

# Multiple-Channel Representation Readout
class MultipleChannelRepresentation(nn.Cell):
    def __init__(self,
        ninteractions,
        n_atom_basis,
        n_hidden=0,
        activation=Swish(),
    ):
        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.ninteractions = ninteractions

        sub_dim = n_atom_basis // ninteractions
        last_dim = n_atom_basis - (sub_dim * (ninteractions - 1))
        sub_dims = [sub_dim for _ in range(ninteractions - 1)]
        sub_dims.append(last_dim)

        if n_hidden > 0:
            hidden_layers = [n_atom_basis for _ in range(n_hidden)]
            self.mcr = nn.CellList([
                MLP(n_atom_basis,sub_dims[i],hidden_layers,activation=activation)
                for i in range(ninteractions)
                ])
        else:
            self.mcr = nn.CellList([
                Dense(n_atom_basis,sub_dims[i],activation=activation)
                for i in range(ninteractions)
                ])

        self.concat = P.Concat(-1)
        self.reduce_sum = P.ReduceSum()

    def construct(self,xlist,atom_mask=None):
        Xt = ()
        for i in range(self.ninteractions):
            Xt = Xt + (self.mcr[i](xlist[i]),)
        return self.concat(Xt)

class TensorSum(nn.Cell):
    def __init__(self,):
        super().__init__()

    def construct(self,xlist,atom_mask=None):
        xs = 0
        for x in xlist:
            xs = xs + x
        return xs

class GatherNeighbors(nn.Cell):
    def __init__(self,dim,fixed_neigh=False):
        super().__init__()
        self.fixed_neigh = fixed_neigh

        self.broad_ones = P.Ones()((1,1,dim),ms.int32)

        if fixed_neigh:
            self.gatherd = None
        else:
            self.gatherd = P.GatherD()

    def construct(self,inputs,neighbors):
        # Construct auxiliary index vector
        ns = neighbors.shape
        
        # Get atomic positions of all neighboring indices

        if self.fixed_neigh:
            return F.gather(inputs,neighbors[0],-2)
        else:
            # [B, A, N] -> [B, A*N, 1]
            neigh_idx = F.reshape(neighbors,(ns[0],ns[1]*ns[2],-1))
            # [B, A*N, V] = [B, A*N, V] * [1, 1, V]
            neigh_idx = neigh_idx * self.broad_ones
            # [B, A*N, V] gather from [B, A, V]
            outputs = self.gatherd(inputs,1,neigh_idx)
            # [B, A, N, V]
            return F.reshape(outputs,(ns[0],ns[1],ns[2],-1))

class Filter(nn.Cell):
    def __init__(self,
        num_rbf,
        dim_filter,
        activation,
        n_hidden=1,
        use_last_activation=False,
    ):
        super().__init__()

        if n_hidden > 0:
            hidden_layers = [dim_filter for _ in range(n_hidden)]
            self.dense_layers = MLP(num_rbf,dim_filter,hidden_layers,activation=activation,use_last_activation=use_last_activation)
        else:
            self.dense_layers = Dense(num_rbf,dim_filter,activation=activation)
        
    def construct(self,rbf):
        return self.dense_layers(rbf)
        
class CFconv(nn.Cell):
    def __init__(
        self,
        num_rbf,
        dim_filter,
        activation,
        ):
        super().__init__()
        # filter block used in interaction block
        self.filter=Filter(num_rbf,dim_filter,activation)
                
    def construct(self,x,f_ij,c_ij=None):
        # n_batch, n_atom, n_nbh, n_rbf = rbf.shape
        # rbf = F.reshape(rbf,(-1,n_rbf))
        W = self.filter(f_ij)
        if c_ij is not None:
            W = W * c_ij
        # W = F.reshape(W,(n_batch,n_atom,n_nbh,-1))
        
        return x * W

class ScaleShift(nn.Cell):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self, mean, stddev):
        super().__init__()
        self.mean = mean
        self.stdev = stddev

    def construct(self, inputs):
        """Compute layer output.

        Args:
            inputs (mindspore.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = inputs * self.stddev + self.mean
        return y


class Standardize(nn.Cell):
    r"""Standardize layer for shifting and scaling.

    .. math::
       y = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division.

    """

    def __init__(self, mean, stddev, eps=1e-9):
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.eps = F.ones_like(stddev) * eps

    def construct(self, inputs):
        """Compute layer output.

        Args:
            inputs (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        # Add small number to catch divide by zero
        y = (inputs - self.mean) / (self.stddev + self.eps)
        return y


class Aggregate(nn.Cell):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis, mean=False):
        super().__init__()
        self.average = mean
        self.axis = axis
        # ~ self.keepdim = keepdim
        self.reduce_sum=P.ReduceSum()
        self.maximum=P.Maximum()

    def construct(self, inputs, mask=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            inputs = inputs * F.expand_dims(mask,-1)
        # compute sum of input along axis

        y = self.reduce_sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = self.reduce_sum(mask, self.axis)
                N = self.maximum(N, other=F.ones_like(N))
            else:
                N = inputs.shape[self.axis]
                
            y = y / N
        return y

class PositionalEmbedding(nn.Cell):
    def __init__(self,dim):
        super().__init__()

        self.layer_norm = nn.LayerNorm((dim,),-1,-1)

        self.xg2q = Dense(dim,dim,has_bias=False)
        self.xg2k = Dense(dim,dim,has_bias=False)
        self.xg2v = Dense(dim,dim,has_bias=False)

        self.mul = P.Mul()
        self.concat = P.Concat(-2)
    
    def construct(self,xi,g_ii,xij,g_ij,t=0,c_ij=None):
        r"""Get query, key and query from atom types and positions

        Args:
            xi   (Mindspore.Tensor [B, A, V]):
            g_ii (Mindspore.Tensor [B, A, V]):
            xij  (Mindspore.Tensor [B, A, N, V]):
            g_ij (Mindspore.Tensor [B, A, N, V]):
            t    (Mindspore.Tensor [V]):

        Marks:
            B:  Batch size
            A:  Number of atoms
            N:  Number of neighbor atoms
            N': Number of neighbor atoms and itself (N' = N + 1)
            V:  Dimensions of atom embedding (V = v * h)

        Returns:
            query  (Mindspore.Tensor [B, A, 1, V]):
            key    (Mindspore.Tensor [B, A, N', V]):
            value  (Mindspore.Tensor [B, A, N', V]):

        """
        # [B, A, V] * [B, A, V] = [B, A, V]
        xgii = self.mul(xi,g_ii)
        # [B, A, N, V] * [B, A, N, V] = [B, A, N, V]
        xgij = self.mul(xij,g_ij)

        # [B, A, 1, V]
        xgii = F.expand_dims(xgii,-2)
        # [B, A, N', V]
        xgij = self.concat((xgii,xgij))
        if c_ij is not None:
            # [B, A, N', V] * [B, A, N', 1]
            xgij = xgij * F.expand_dims(c_ij,-1)
        
        xgii = self.layer_norm(xgii + t)
        xgij = self.layer_norm(xgij + t)

        # [B, A, 1, V]
        query = self.xg2q(xgii)
        # [B, A, N', V]
        key   = self.xg2k(xgij)
        # [B, A, N', V]
        value = self.xg2v(xgij)

        return query, key, value

class MultiheadAttention(nn.Cell):
    r"""Compute multi-head attention.

    Args:
        dim     (int): Demension of atom embedding (V)
        n_heads (int): Number of heads (h)

    Marks:
        B:  Batch size
        A:  Number of atoms
        N': Number of neighbor atoms and itself
        V:  Dimensions of atom embedding
        h:  Number of heads for multi-head attention
        v:  Dimensions per head (V = v * h)

    """
    def __init__(self,dim,n_heads):
        super().__init__()

        # h
        self.n_heads = n_heads

        # v = V / h
        self.size_per_head = dim // n_heads
        scores_mul = 1.0 / np.sqrt(float(self.size_per_head))
        self.scores_mul = ms.Tensor(scores_mul,ms.float32)

        self.exones = P.Ones()((1,1,n_heads,1,1),ms.int32)

        # shape = (h, v)
        self.reshape_tail = (self.n_heads, self.size_per_head)

        self.output = Dense(dim,dim,has_bias=False)

        self.mul = P.Mul()
        self.div = P.Div()
        self.softmax = P.Softmax()
        self.bmm = P.BatchMatMul()
        self.bmmt = P.BatchMatMul(transpose_b=True)
        self.squeeze = P.Squeeze(-2)
        self.reducesum = P.ReduceSum(keep_dims=True)
        
        self.transpose = P.Transpose()
        self.trans_shape = (0, 1, 3, 2, 4)
    
    def construct(self,query,key,value,cutoff=None,mask=None):
        r"""Compute multi-head attention.

        Args:
            query  (Mindspore.Tensor [B, A, 1, V]):
            key    (Mindspore.Tensor [B, A, N', V]):
            value  (Mindspore.Tensor [B, A, N', V]):
            cutoff (Mindspore.Tensor [B, A, 1, N'] or [B, A, 1, 1, N']):

        Returns:
            Mindspore.Tensor [B, A, V]: multi-head attention output.

        """
        if self.n_heads > 1:
            q_reshape = query.shape[:-1] + self.reshape_tail
            k_reshape = key.shape[:-1]   + self.reshape_tail
            v_reshape = value.shape[:-1] + self.reshape_tail

            # [B, A, 1, h, v]
            Q = F.reshape(query,q_reshape)
            # [B, A, h, 1, v]
            Q = self.transpose(Q,self.trans_shape)

            # [B, A, N', h, v]
            K = F.reshape(key,k_reshape)
            # [B, A, h, N', v]
            K = self.transpose(K,self.trans_shape)

            # [B, A, N', h, v]
            V = F.reshape(value,v_reshape)
            # [B, A, h, N', v]
            V = self.transpose(V,self.trans_shape)

            # [B, A, h, 1, v] x [B, A, h, N', v]^T / \sqrt(v)
            # [B, A, h, 1, v] x [B, A, h, v, N'] = [B, A, h, 1, N']
            attention_scores = self.bmmt(Q,K)
            attention_scores = self.mul(attention_scores,self.scores_mul)

            if cutoff is None:
                attention_probs = self.softmax(attention_scores)
            else:
                # [B, A, 1, 1, N']
                exmask = F.expand_dims(F.expand_dims(mask,-2),-2)
                # [B, A, h, 1, N']
                mhmask = exmask * self.exones
                large_neg = F.ones_like(attention_scores) * -5e4
                attention_scores = F.select(mhmask>0,attention_scores,large_neg)
                attention_probs = self.softmax(attention_scores)
                excut  = F.expand_dims(F.expand_dims(cutoff,-2),-2)
                # [B, A, h, 1, N'] * [B, A, 1, 1, N']
                attention_probs = self.mul(attention_probs,excut)

            # [B, A, h, 1, N'] x [B, A, h, N', v] = [B, A, h, 1, v]
            context = self.bmm(attention_probs,V)
            # [B, A, 1, h, v]
            context = self.transpose(context,self.trans_shape)
            # [B, A, 1, V]
            context = F.reshape(context,query.shape)
        else:
            # [B, A, 1, V] x [B, A, N', V]^T / \sqrt(V)
            # [B, A, 1, V] x [B, A, V, N'] = [B, A, 1, N']
            attention_scores = self.bmmt(query,key) * self.scores_mul
            
            if cutoff is None:
                attention_probs = self.softmax(attention_scores)
            else:
                large_neg = F.ones_like(attention_scores) * -5e4
                attention_scores = F.select(mask,attention_scores,large_neg)
                attention_probs = self.softmax(attention_scores)
                # [B, A, 1, N'] * [B, A, 1, N']
                attention_probs = attention_probs * F.expand_dims(cutoff,-2)
            
            # [B, A, 1, N'] x [B, A, N', V] = [B, A, 1, V]
            context = self.bmm(attention_probs,value)

        # [B, A, V]
        context = self.squeeze(context)

        return self.output(context)

class Pondering(nn.Cell):
    def __init__(self,n_in,n_hidden=0,bias_const=1.):
        super().__init__()

        if n_hidden == 0:
            self.dense = nn.Dense(n_in,1,has_bias=True,weight_init='xavier_uniform',bias_init=Constant(bias_const),activation='sigmoid',)
        elif n_hidden > 0:
            nets=[]
            for i in range(n_hidden):
                nets.append(nn.Dense(n_in, n_in, weight_init='xavier_uniform', activation='relu'))
            nets.append(nn.Dense(n_in, 1, bias_init=Constant(bias_const), activation='sigmoid'))
            self.dense = nn.SequentialCell(nets)
        else:
            raise ValueError("n_hidden cannot be negative!")

        self.squeeze = P.Squeeze(-1)

    def construct(self,x):
        y = self.dense(x)
        return self.squeeze(y)

# Modified from:
# https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/UTransformer.py
class ACTWeight(nn.Cell):
    def __init__(self,shape,threshold=0.9):
        super().__init__()
        self.threshold = threshold

        self.zeros_like = P.ZerosLike()
        self.ones_like = P.OnesLike()
        # self.select = P.Select()

    def construct(self,prob,halting_prob,n_updates):
        # zeros = self.zeros_like(halting_prob)
        # ones = self.ones_like(halting_prob)

        # Mask for inputs which have not halted last cy
        running = F.cast(halting_prob < 1.0, ms.float32)
        # running = self.select(halting_prob < 1.0,ones,zeros)

        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        add_prob = prob * running
        new_prob = halting_prob + add_prob 
        mask_run = F.cast(new_prob <= self.threshold,ms.float32)
        mask_halt = F.cast(new_prob > self.threshold,ms.float32)
        # mask_run = self.select(new_prob <= self.threshold,ones,zeros)
        # mask_halt = self.select(new_prob > self.threshold,ones,zeros)

        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = mask_run * running
        running_prob = halting_prob + prob * still_running

        # Mask of inputs which halted at this step
        new_halted = mask_halt * running

        # Compute remainders for the inputs which halted at this step
        remainders = new_halted * (1.0 - running_prob)

        # Add the remainders to those inputs which halted at this step
        # halting_prob = new_prob + remainders
        dp = add_prob + remainders

        # Increment n_updates for all inputs which are still running
        # n_updates = n_updates + running
        dn = running

        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # prob when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = prob * still_running + new_halted * remainders
        w = F.expand_dims(update_weights,-1)

        return w, dp, dn

class Num2Mask(nn.Cell):
    def __init__(self,dim):
        super().__init__()
        self.range = nn.Range(dim)
        ones = P.Ones()
        self.ones = ones((dim),ms.int32)
    def construct(self,num):
        nmax = num * self.ones
        idx = F.ones_like(num) * self.range()
        return idx < nmax

class Number2FullConnectNeighbors(nn.Cell):
    def __init__(self,tot_atoms):
        super().__init__()
        # tot_atoms: A
        # tot_neigh: N =  A - 1
        tot_neigh = tot_atoms -1
        arange = nn.Range(tot_atoms)
        nrange = nn.Range(tot_neigh)
        
        self.ones = P.Ones()
        self.aones = self.ones((tot_atoms),ms.int32)
        self.nones = self.ones((tot_neigh),ms.int32)
        
        # neighbors for no connection (A*N)
        # [[0,0,...,0],
        #  [1,1,...,1],
        #  ...........,
        #  [N,N,...,N]]
        self.nnc = F.expand_dims(arange(),-1) * self.nones
        # copy of the index range (A*N)
        # [[0,1,...,N-1],
        #  [0,1,...,N-1],
        #  ...........,
        #  [0,1,...,N-1]]
        crange = self.ones((tot_atoms,1),ms.int32) * nrange()
        # neighbors for full connection (A*N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        self.nfc = crange + F.cast(self.nnc <= crange,ms.int32)
        
        crange1 = crange + 1
        # the matrix for index range (A*N)
        # [[1,2,3,...,N],
        #  [1,2,3,...,N],
        #  [2,2,3,....N],
        #  [3,3,3,....N],
        #  .............,
        #  [N,N,N,...,N]]
        self.mat_idx = F.select(crange1>self.nnc,crange1,self.nnc)

    def get_full_neighbors(self):
        return F.expand_dims(self.nfc,0)
        
    def construct(self,num_atoms):
        # broadcast atom numbers to [B*A*N]
        # a_i: number of atoms in each molecule
        # [[a_0]*A*N,[a_1]*A*N,...,[a_N]*A*N]]
        exnum = num_atoms * self.aones
        exnum = F.expand_dims(exnum,-1) * self.nones

        # [B,1,1]
        exones = self.ones((num_atoms.shape[0],1,1),ms.int32)
        # broadcast to [B*A*N]: [B,1,1] * [1,A,N]
        exnfc = exones * F.expand_dims(self.nfc,0)
        exnnc = exones * F.expand_dims(self.nnc,0)
        exmat = exones * F.expand_dims(self.mat_idx,0)

        mask = exmat < exnum

        neighbors = F.select(mask,exnfc,exnnc)

        return neighbors,mask

class Types2FullConnectNeighbors(nn.Cell):
    def __init__(self,tot_atoms):
        super().__init__()
        # tot_atoms: A
        # tot_neigh: N =  A - 1
        tot_neigh = tot_atoms -1
        arange = nn.Range(tot_atoms)
        nrange = nn.Range(tot_neigh)
        
        self.ones = P.Ones()
        self.aones = self.ones((tot_atoms),ms.int32)
        self.nones = self.ones((tot_neigh),ms.int32)
        self.eaones = F.expand_dims(self.aones,-1)
        
        # neighbors for no connection (A*N)
        # [[0,0,...,0],
        #  [1,1,...,1],
        #  ...........,
        #  [N,N,...,N]]
        self.nnc = F.expand_dims(arange(),-1) * self.nones

        # copy of the index range (A*N)
        # [[0,1,...,N-1],
        #  [0,1,...,N-1],
        #  ...........,
        #  [0,1,...,N-1]]
        exrange = self.ones((tot_atoms,1),ms.int32) * nrange()

        # neighbors for full connection (A*N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        self.nfc = exrange + F.cast(self.nnc <= exrange,ms.int32)
        
        self.ar0 = nn.Range(0,tot_neigh)()
        self.ar1 = nn.Range(1,tot_atoms)()

    def get_full_neighbors(self):
        return F.expand_dims(self.nfc,0)
        
    def construct(self,atom_types):
        # [B,1,1]
        exones = self.ones((atom_types.shape[0],1,1),ms.int32)
        # broadcast to [B*A*N]: [B,1,1] * [1,A,N]
        exnfc = exones * F.expand_dims(self.nfc,0)
        exnnc = exones * F.expand_dims(self.nnc,0)
        
        tmask = F.select(atom_types>0, F.ones_like(atom_types), F.ones_like(atom_types) * -1)
        tmask = F.cast(tmask,ms.float32)
        extmask = F.expand_dims(tmask,-1) * self.nones
        
        mask0 = F.gather(tmask,self.ar0,-1)
        mask0 = F.expand_dims(mask0,-2) * self.eaones
        mask1 = F.gather(tmask,self.ar1,-1)
        mask1 = F.expand_dims(mask1,-2) * self.eaones      
        
        mtmp = F.select(exnfc > exnnc, mask1, mask0)
        mask  = F.select(extmask > 0, mtmp, F.ones_like(mtmp) * -1)
        mask  = mask > 0
        
        idx = F.select(mask, exnfc, exnnc)

        return idx,mask

