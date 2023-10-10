import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from airnetpack.base import GatherNeighbors

class AtomDistances(nn.Cell):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        neighbors_fixed (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self,fixed_atoms=False,dim=3):
        super().__init__()
        self.fixed_atoms=fixed_atoms
        self.reducesum = P.ReduceSum()
        self.pow = P.Pow()
        # self.concat = P.Concat()
        # self.pack = P.Pack()
        self.gatherd = P.GatherD()
        self.norm = nn.Norm(-1)

        self.gather_neighbors = GatherNeighbors(dim,fixed_atoms)

    def construct(
        self, positions, neighbors, neighbor_mask=None, cell=None, cell_offsets=None
        ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (ms.Tensor[float]): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (ms.Tensor[int]): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) or (N_at x N_nbh) shape.
            cell (ms.tensor[float], optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (ms.Tensor[float], optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (ms.Tensor[bool], optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            ms.Tensor[float]: layer output of (N_b x N_at x N_nbh) shape.

        """

        pos_xyz = self.gather_neighbors(positions,neighbors)

        # Subtract positions of central atoms to get distance vectors
        dist_vec = pos_xyz - F.expand_dims(positions,-2)

        # distances = self.norm(dist_vec)
        distances = F.square(dist_vec)
        distances = self.reducesum(distances,-1)
        distances = self.pow(distances,0.5)
        
        if neighbor_mask is not None:
            distances = F.select(neighbor_mask,distances,F.ones_like(distances)*999)

        return distances
