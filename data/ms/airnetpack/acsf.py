import math
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.api import ms_function

from airnetpack.cutoff import CosineCutoff

__all__ = [
    "GaussianSmearing",
    "LogGaussianDistribution",
]

# radial_filter in RadialDistribution
class GaussianSmearing(nn.Cell):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self, d_min=0.0, d_max=5.0, num_rbf=32, sigma=None, centered=False, trainable=False
    ):
        super().__init__()
        # compute offset and width of Gaussian functions
        offset = Tensor(np.linspace(d_min, d_max, num_rbf),ms.float32)

        if sigma is None:
            sigma = (d_max-d_min) / (num_rbf-1)

        width = sigma * F.ones_like(offset)
        
        self.width = width
        self.offset = offset
        self.centered = centered
        
        if trainable:
            self.width = ms.Parameter(width,"widths")
            self.offset = ms.Parameter(offset,"offset")

    def construct(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        ex_dis=F.expand_dims(distances,-1)
        if not self.centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / F.square(self.width)
            # Use advanced indexing to compute the individual components
            # ~ diff = distances[:, :, :, None] - offset[None, None, None, :]
            ex_offset=F.reshape(self.offset,(1,1,1,-1))
            diff = ex_dis - ex_offset
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / F.square(self.offset)
            # if Gaussian functions are centered, no offset is subtracted
            diff = ex_dis
        # compute smear distance values
        exp = P.Exp()
        gauss = exp(coeff * F.square(diff))
        return gauss
        
class LogGaussianDistribution(nn.Cell):
    def __init__(
        self,
        d_min=1e-3,
        d_max=1.0,
        num_rbf=32,
        sigma=None,
        trainable=False,
        min_cutoff=False,
        max_cutoff=False,
    ):
        super().__init__()
        if d_max <= d_min:
            raise ValueError('The argument "d_max" must be larger'+
                'than the argument "d_min" in LogGaussianDistribution!')
            
        if d_min <= 0:
            raise ValueError('The argument "d_min" must be '+
                ' larger than 0 in LogGaussianDistribution!')
            
        self.d_max = d_max
        self.d_min = d_min / d_max
        self.min_cutoff=min_cutoff
        self.max_cutoff=max_cutoff
        
        self.log = P.Log()
        self.exp = P.Exp()
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.zeroslike = P.ZerosLike()
        self.oneslike = P.OnesLike()

        # linspace = nn.LinSpace(log_dmin,0,n_gaussians)
        
        log_dmin=math.log(self.d_min)
        # self.centers = linspace()
        # self.ones = self.oneslike(self.centers)
        centers = np.linspace(log_dmin,0,num_rbf)
        self.centers = Tensor(centers,ms.float32)
        ones = np.ones_like(centers)
        self.ones = Tensor(ones,ms.float32)
        
        if sigma is None:
            sigma = -log_dmin / (num_rbf-1)
        self.rescale = -0.5 / (sigma * sigma)

    def construct(self, distance):
        dis = distance / self.d_max
        
        if self.min_cutoff:
            dis = self.max(dis,self.d_min)

        exdis = F.expand_dims(dis,-1)
        rbfdis = exdis * self.ones
        
        log_dis = self.log(rbfdis)
        log_diff = log_dis - self.centers
        log_diff2 = F.square(log_diff)
        log_gauss = self.exp( self.rescale * log_diff2  )

        if self.max_cutoff:
            ones = self.onesslike(exdis)
            zeros = self.zeroslike(exdis)
            cuts = F.select(exdis < 1.0, ones, zeros)
            log_gauss = log_gauss * cuts
        
        return log_gauss
