import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from airnetpack.interactions import SchNetInteraction,AirNetInteraction
from airnetpack.base import AtomwiseReadout,Filter,Aggregate
from airnetpack.base import MultipleChannelRepresentation,TensorSum
from airnetpack.base import Types2FullConnectNeighbors
from airnetpack.cutoff import CosineCutoff,SmoothCutoff
from airnetpack.acsf import GaussianSmearing,LogGaussianDistribution
from airnetpack.activations import ShiftedSoftplus,Swish
from airnetpack.neighbors import AtomDistances
from airnetpack import Properties

class GNN_Model(nn.Cell):
    r"""Basic class for graph neural network (GNN) based deep molecular model

    Args:
        num_atomtypes (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_atomembedding (int): dimension of the vectors for atomic embedding
        atom_types (ms.Tensor[int], optional): atomic index 
        distance_expansion(nn.Cell, optional): the alghorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.

    """

    def __init__(
        self,
        num_atomtypes,
        dim_atomembedding,
        min_rbf_dis,
        max_rbf_dis,
        num_rbf,
        output_dim=1,
        rbf_sigma=None,
        trainable_rbf=False,
        distance_expansion=None,
        cutoff=None,
        cutoff_network=None,
        rescale_rbf=False,
        use_all_interactions=False,
    ):
        super().__init__()
        self.num_atomtypes = num_atomtypes
        self.dim_atomembedding = dim_atomembedding
        self.num_rbf = num_rbf
        self.distance_expansion = distance_expansion
        self.rescale_rbf = rescale_rbf
        self.output_dim = output_dim
        # ~ self.n_interactions=n_interactions
        
        self.network_name='GNN_Model'

        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size dim_atomembedding
        self.embedding = nn.Embedding(num_atomtypes, dim_atomembedding, use_one_hot=True, embedding_table=Normal(1.0))

        self.filter = None
        
        self.fixed_atoms=False

        # layer for expanding interatomic distances in a basis
        if distance_expansion is not None:
            self.distance_expansion = distance_expansion(
                d_min = min_rbf_dis, d_max = max_rbf_dis, num_rbf = num_rbf, sigma=rbf_sigma, trainable = trainable_rbf
            )
        else:
            self.distance_expansion = None

        if cutoff_network is None:
            self.cutoff_network = None
            self.cutoff = None
        else:
            if cutoff is None:
                self.cutoff_network = cutoff_network(max_rbf_dis)
                self.cutoff = max_rbf_dis
            else:
                self.cutoff_network = cutoff_network(cutoff)
                self.cutoff = cutoff
        
        self.interactions = None
        
        self.readout = None
        self.use_all_interactions = use_all_interactions
        self.gather_interactions = None

        self.debug_fun = None

        self.ones = P.Ones()

        # self.broadcast_to = P.BroadcastTo((Properties.batch_size,Properties.atom_number,Properties.basis_number))

    def _set_fixed_atoms(self,fixed_atoms=True):
        self.fixed_atoms = fixed_atoms
        
    def _set_fixed_neighbors(self):
        for interaction in self.interactions:
            interaction._set_fixed_neighbors(True)

    def _get_cutoff(self,r_ij,neighbor_mask):
        return self.cutoff_network(r_ij,neighbor_mask)

    def _get_rbf(self,dis):
        # expand interatomic distances (for example, Gaussian smearing)
        if self.distance_expansion is None:
            rbf = F.expand_dims(dis,-1)
        else:
            rbf = self.distance_expansion(dis)
        
        if self.rescale_rbf:
            rbf = rbf * 2.0 - 1.0

        if self.filter is not None:
            return self.filter(rbf)
        else:
            return rbf

    def _get_self_rbf(self):
        return 0

    def construct(self, r_ij, atom_types=None, neighbors=None, neighbor_mask=None):
        """Compute interaction output.

        Args:
            r_ij (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (ms.Tensor[bool], optional): mask to filter out non-existing neighbors
                introduced via padding.
            atom_types (ms.Tensor[int], optional): atomic index 

        Returns:
            torch.Tensor: block output with (N_b, N_a, N_basis) shape.

        """

        if self.fixed_atoms:
            e = self.ones((r_ij.shape[0],1,1),ms.float32) * self.embedding(atom_types)
        else:
            e = self.embedding(atom_types)
        
        f_ij = self._get_rbf(r_ij)
        f_ii = self._get_self_rbf()

        # apply cutoff
        if self.cutoff_network is None:
            c_ij = 1
            mask = neighbor_mask
        else:
            c_ij,mask = self._get_cutoff(r_ij,neighbor_mask)
        
        debug_info = []
        # continuous-filter convolution interaction block followed by Dense layer
        x = e
        n_interactions = len(self.interactions)
        xlist = []
        for i in range(n_interactions):
            x, info = self.interactions[i](x, e, f_ii, f_ij, c_ij, neighbors, mask)
            debug_info.append(info)
            if x is None:
                return None
            if self.use_all_interactions:
                xlist.append(x)

        if self.use_all_interactions and self.gather_interactions is not None:
            x = self.gather_interactions(xlist,mask)

        if self.debug_fun is not None:
            self.debug_fun(debug_info)

        if self.readout is not None:
            y = self.readout(x,mask)
        else:
            y = x

        return y
        
class SchNet(GNN_Model):
    r"""SchNet architecture for learning representations of atomistic systems.
    
    References:
     .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
        Quantum-chemical insights from deep tensor neural networks.
        Nature Communications, 8, 13890. 2017.
     .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
        SchNet: A continuous-filter convolutional neural network for modeling quantum
        interactions.
        In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
     .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
        SchNet - a deep learning architecture for molceules and materials.
        The Journal of Chemical Physics 148 (24), 241722. 2018.

    Args:
        
        num_atomtypes (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_atomembedding (int): dimension of the vectors for atomic embedding
        dim_filter (int): dimension of the vectors for filters used in continuous-filter convolution.
        n_interactions (int, optional): number of interaction blocks.
        max_distance (float): the maximum distance to calculate RBF.
        atom_types (ms.Tensor[int], optional): atomic index 
        distance_expansion(nn.Cell, optional): the alghorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.

    """
    def __init__(
        self,
        num_atomtypes=100,
        dim_atomembedding=64,
        min_rbf_dis=0,
        max_rbf_dis=0.5,
        num_rbf=32,
        dim_filter=64,
        n_interactions=3,
        activation=ShiftedSoftplus(),
        output_dim=1,
        rbf_sigma=None,
        distance_expansion=GaussianSmearing,
        cutoff=None,
        cutoff_network=CosineCutoff,
        normalize_filter=False,
        coupled_interactions=False,
        trainable_rbf=False,
    ):
        super().__init__(
            num_atomtypes=num_atomtypes,
            dim_atomembedding=dim_atomembedding,
            min_rbf_dis=min_rbf_dis,
            max_rbf_dis=max_rbf_dis,
            num_rbf=num_rbf,
            output_dim=output_dim,
            rbf_sigma=rbf_sigma,
            distance_expansion=distance_expansion,
            cutoff = cutoff,
            cutoff_network=cutoff_network,
            rescale_rbf=False,
            use_all_interactions=False,
            trainable_rbf=trainable_rbf,
            )
        self.network_name = 'SchNet'

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        n_input=dim_atomembedding,
                        num_rbf=num_rbf,
                        dim_filter=dim_filter,
                        activation=activation,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        n_input=dim_atomembedding,
                        num_rbf=num_rbf,
                        dim_filter=dim_filter,
                        activation=activation,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )
        
        outdim = dim_atomembedding // 2
        # readout layer
        self.readout = AtomwiseReadout(dim_atomembedding,self.output_dim,[outdim,],activation)
        
class AirNet(GNN_Model):
    r"""SchNet architecture for learning representations of atomistic systems.
    
    References:
     .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
        Quantum-chemical insights from deep tensor neural networks.
        Nature Communications, 8, 13890. 2017.
     .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
        SchNet: A continuous-filter convolutional neural network for modeling quantum
        interactions.
        In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
     .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
        SchNet - a deep learning architecture for molceules and materials.
        The Journal of Chemical Physics 148 (24), 241722. 2018.

    Args:
        
        num_atomtypes (int): maximum number of atomic types
        num_rbf (int): number of the serial of radical basis functions (RBF)
        dim_atomembedding (int): dimension of the vectors for atomic embedding
        dim_filter (int): dimension of the vectors for filters used in continuous-filter convolution.
        n_interactions (int, optional): number of interaction blocks.
        max_distance (float): the maximum distance to calculate RBF.
        atom_types (ms.Tensor[int], optional): atomic index 
        distance_expansion(nn.Cell, optional): the alghorithm to calculate RBF
        cutoff_network (nn.Cell, optional): the algorithm to calculate cutoff.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.

    """
    def __init__(
        self,
        num_atomtypes=100,
        dim_atomembedding=64,
        min_rbf_dis=0.05,
        max_rbf_dis=1,
        num_rbf=32,
        n_interactions=3,
        n_heads=8,
        max_cycles=10,
        activation=Swish(),
        output_dim=1,
        self_dis=None,
        rbf_sigma=None,
        distance_expansion=LogGaussianDistribution,
        cutoff=None,
        cutoff_network=SmoothCutoff,
        public_filter=True,
        coupled_interactions=False,
        trainable_gaussians=False,
        use_pondering=True,
        fixed_cycles=False,
        rescale_rbf=True,
        use_time_embedding=True,
        use_all_interactions=True,
        use_mcr=False,
        debug=False,
    ):
        super().__init__(
            num_atomtypes=num_atomtypes,
            dim_atomembedding=dim_atomembedding,
            min_rbf_dis=min_rbf_dis,
            max_rbf_dis=max_rbf_dis,
            num_rbf=num_rbf,
            output_dim=output_dim,
            rbf_sigma=rbf_sigma,
            distance_expansion=distance_expansion,
            cutoff=cutoff,
            cutoff_network=cutoff_network,
            rescale_rbf=rescale_rbf,
            use_all_interactions=use_all_interactions,
            )
        self.network_name = 'AirNet'
        self.max_distance = max_rbf_dis
        self.min_distance = min_rbf_dis

        if self_dis is None:
            self.self_dis = self.min_distance
        else:
            self.self_dis = self_dis

        self.self_dis_tensor = Tensor([self.self_dis],ms.float32)

        self.n_heads = n_heads
        
        if use_time_embedding:
            time_embedding = self._get_time_signal(max_cycles,dim_atomembedding)
        else:
            time_embedding = [ 0 for _ in range(max_cycles) ]
        
        if public_filter:
            inter_filter = False
            self.filter = Filter(num_rbf,dim_atomembedding,None)
        else:
            inter_filter = True
            self.filter = None

        self.n_interactions = n_interactions

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    AirNetInteraction(
                        dim_atom_embed=dim_atomembedding,
                        num_rbf=num_rbf,
                        n_heads=n_heads,
                        activation=activation,
                        max_cycles=max_cycles,
                        time_embedding=time_embedding,
                        use_filter=inter_filter,
                        use_pondering=use_pondering,
                        fixed_cycles=fixed_cycles,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    AirNetInteraction(
                        dim_atom_embed=dim_atomembedding,
                        num_rbf=num_rbf,
                        n_heads=n_heads,
                        activation=activation,
                        max_cycles=max_cycles,
                        time_embedding=time_embedding,
                        use_filter=inter_filter,
                        use_pondering=use_pondering,
                        fixed_cycles=fixed_cycles,
                    )
                    for i in range(n_interactions)
                ]
            )

        # readout layer
        if self.use_all_interactions and n_interactions > 1:
            if use_mcr:
                self.gather_interactions = MultipleChannelRepresentation(n_interactions,dim_atomembedding,1,activation)
            else:
                self.gather_interactions = TensorSum()
        else:
            self.gather_interactions = None
        
        readoutdim = int(dim_atomembedding / 2)
        self.readout = AtomwiseReadout(dim_atomembedding,self.output_dim,[readoutdim,],activation)

        if debug:
            self.debug_fun = self._debug_fun

        self.lmax_label = []
        for i in range(n_interactions):
            self.lmax_label.append('l'+str(i)+'_cycles')

        self.fill = P.Fill()
        self.concat = P.Concat(-1)
        self.pack = P.Pack(-1)
        self.reducesum = P.ReduceSum()
        self.reducemax = P.ReduceMax()
        self.tensor_summary = P.TensorSummary()
        self.scalar_summary = P.ScalarSummary()

    def _get_cutoff(self,r_ij,neighbor_mask):
        rii_shape = r_ij.shape[:-1] + (1,)
        # [B, A, 1]
        r_ii = self.fill(ms.float32,rii_shape,self.self_dis)
        # [B, A, N']s
        return self.cutoff_network(self.concat((r_ii,r_ij)),neighbor_mask)

    def _get_self_rbf(self):
        return self._get_rbf(self.self_dis_tensor)

    def _get_time_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):

        """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py
        """
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = ( np.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                        'constant', constant_values=[0.0, 0.0])

        return Tensor(signal,ms.float32)

    def _debug_fun(self,info):
        
        for i,l in zip(info,self.lmax_label):
            cycles = self.reducemax(i)
            self.scalar_summary(l,cycles)

        n_updates = self.pack(info)
        updates_sum = self.reducesum(n_updates,-1)
        updates_max = self.reducemax(updates_sum,0)
        self.tensor_summary('update_numbers',updates_max)
        max_cycles = self.reducemax(updates_max)
        self.scalar_summary('max_cycles',max_cycles)
        return max_cycles
        

class MolCalculator(nn.Cell):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        dim_atomembedding (int, optional): dimension of the vectors to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        dim_filter (int, optional): number of filters used in continuous-filter convolution
        
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        charged_systems (bool, optional):

    """

    def __init__(
        self,
        model,
        scale=1.0,
        shift=0.0,
        max_atoms_num=0,
        aggregate=True,
        average=False,
        atom_types=None,
        full_connect=False,
    ):
        super().__init__()
        
        self.predict=model
        # dim_atomembedding=model.dim_atomembedding
        self.full_connect=full_connect
        
        self.scale = scale
        self.shift = shift

        self.aggregate = aggregate
        self.average = average

        self.reducesum = P.ReduceSum(keep_dims=False)
        self.molsum = P.ReduceSum(keep_dims=True)
        self.reducemean = P.ReduceMean(keep_dims=False)
        
        if atom_types is None:
            self.fixed_atoms=False
            self.num_atoms=0
        else:
            self.fixed_atoms=True
            model._set_fixed_atoms(True)

            if len(atom_types.shape) == 1:
                self.num_atoms=len(atom_types)
            elif len(atom_types.shape) == 2:
                self.num_atoms=len(atom_types[0])

            if self.num_atoms <= 0:
                raise ValueError("The 'num_atoms' cannot be 0 "+
                    "'atom_types' is not 'None' in MolCalculator!")

            if type(atom_types) is not Tensor:
                atom_types = Tensor(atom_types,ms.int32)

            self.atom_types = atom_types

        self.neighbors = None
        self.mask = None
        self.fc_neighbors = None
        if self.full_connect:
            if self.fixed_atoms:
                self.fc_neighbors = Types2FullConnectNeighbors(self.num_atoms)
                self.neighbors = self.fc_neighbors.get_full_neighbors()
            else:
                if max_atoms_num <= 0:
                    raise ValueError("The 'max_atoms_num' cannot be 0 "+
                        "when the 'full_connect' flag is 'True' and " +
                        "'atom_types' is 'None' in MolCalculator!")
                self.fc_neighbors = Types2FullConnectNeighbors(max_atoms_num)

        if self.fixed_atoms and self.full_connect:
            self.distances = AtomDistances(True)
            model._set_fixed_neighbors()
        else:
            self.distances = AtomDistances(False)

        self.ones = P.Ones()

    def construct(self,
        positions,
        atom_types=None,
        neighbors=None,
        neighbor_mask=None,
        ):
        """Compute atomic representations/embeddings.

        Args:
            positions (ms.Tensor[float]): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (ms.Tensor[int], optional): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            neighbor_mask (ms.Tensor[bool], optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.
            atom_types (ms.Tensor[int], optional): atomic indices with 
                (N_b x N_at) shape

        Returns:
            ms.Tensor[float]: prediction for the properties of the molecules
                with (N_b x N_prop) shape

        """

        # compute interatomic distance of every atom to its neighbors
        if atom_types is None:
            if self.fixed_atoms:
                # atom_types = self.ones((positions.shape[0],1),ms.int32) * self.atom_types
                atom_types = self.atom_types
                if  self.full_connect:
                    # neighbors = self.ones((positions.shape[0],1,1),ms.int32) * self.neighbors
                    neighbors = self.neighbors
                    neighbor_mask = None
            else:
                print('Then input term "atom_types" cannot be "None"'+
                'when the "fixed_atoms" flag is "False" in MolCalculator!')
                return None

        if neighbors is None:
            if self.full_connect:
                neighbors,neighbor_mask=self.fc_neighbors(atom_types)
            else:
                # raise RuntimeError('Then input term "neighbors" must be'+
                # ' given when the "full_connect" flag is "False" in MolCalculator!')
                print('Then input term "neighbors" cannot be "None"'+
                'when the "full_connect" flag is "False" in MolCalculator!')
                return None

        r_ij = self.distances(positions,neighbors,neighbor_mask,None,None)

        if r_ij is None:
            print("Please check the error information!")
            return None

        atoms_output = self.predict(r_ij,atom_types,neighbors,neighbor_mask)

        if self.aggregate or self.average:
            if self.aggregate:
                if self.average:
                    mol_output = self.reducemean(atoms_output,-2)
                else:
                    mol_output = self.reducesum(atoms_output,-2)
            else:
                sum_output = self.molsum(atoms_output,-2)
                mol_output = atoms_output / sum_output
        else:
            mol_output = atoms_output
        
        return mol_output * self.scale + self.shift