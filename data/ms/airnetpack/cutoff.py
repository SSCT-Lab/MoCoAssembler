import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

__all__ = ["CosineCutoff", "MollifierCutoff", "HardCutoff", "SmoothCutoff", "get_cutoff_by_string"]


def get_cutoff_by_string(key):
    # build cutoff module
    if key == "hard":
        cutoff_network = HardCutoff
    elif key == "cosine":
        cutoff_network = CosineCutoff
    elif key == "mollifier":
        cutoff_network = MollifierCutoff
    elif key == "smooth":
        cutoff_network = SmoothCutoff
    else:
        raise NotImplementedError("cutoff_function {} is unknown".format(key))
    return cutoff_network


class CosineCutoff(nn.Cell):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.pi = Tensor(np.pi,ms.float32)
        self.cos = P.Cos()
        self.zeros_like = P.ZerosLike()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        
        d_cut = 0.5 * (self.cos(distances * self.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        zeros = self.zeros_like(distances)

        mask = distances < self.cutoff
        if neighbor_mask is not None:
            mask = self.logical_and(mask, neighbor_mask)

        cutoffs = F.select(mask,d_cut,zeros)
        return cutoffs, mask


class MollifierCutoff(nn.Cell):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional): offset added to distances for numerical stability.

    """

    def __init__(self, cutoff=5.0, eps=1.0e-7):
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps
        self.exp = P.Exp()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        mask = ((distances + self.eps) < self.cutoff)
        if neighbor_mask is not None:
            mask = self.logical_and(mask,neighbor_mask)
            
        exponent = 1.0 - 1.0 / (1.0 - F.square(distances * mask / self.cutoff))
        
        cutoffs = self.exp(exponent)
        cutoffs = cutoffs * mask
        return cutoffs, mask


class HardCutoff(nn.Cell):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self, cutoff=5.0):
        super().__init__()

        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        mask = F(distances <= self.cutoff)
        if neighbor_mask is not None:
            self.logical_and(mask,neighbor_mask)

        return F.cast(mask,ms.float32), mask
class SmoothCutoff(nn.Cell):
    r"""Class of smooth cutoff by Ebert, D. S. et al:
        [ref] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003

    ..  math::
        f(r) = 1.0 -  6 * ( r / r_cutoff ) ^ 5
                   + 15 * ( r / r_cutoff ) ^ 4
                   + 10 * ( r / r_cutoff ) ^ 3

    Args:
        d_max (float, optional): the maximum distance (cutoff radius).
        d_min (float, optional): the minimum distance

    """
    def __init__(self,cutoff=1.):
        super().__init__()
            
        self.cutoff=cutoff
        self.zeroslike = P.ZerosLike()
        self.pow = P.Pow()
        self.logical_and = P.LogicalAnd()
        
    def construct(self, distance, neighbor_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor or float): values of interatomic distances.

        Returns:
            mindspore.Tensor or float: values of cutoff function.

        """
        dis = distance / self.cutoff
        cuts =  1.  -  6. * self.pow(dis,5) \
                    + 15. * self.pow(dis,4) \
                    - 10. * self.pow(dis,3)
        zeros = self.zeroslike(distance)
        mask = dis < 1.0
        if neighbor_mask is not None:
            mask = self.logical_and(mask,neighbor_mask)

        cutoff = F.select(mask, cuts, zeros)
        return cutoff, mask
