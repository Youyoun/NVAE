import torch
from torch import nn as nn

from model.auto_encoder import CHANNEL_MULT
from model.neural_ar_operations import ARInvertedResidual, MixLogCDFParam, ELUConv as ARELUConv, mix_log_cdf_flow
from model.neural_operations import get_skip_connection, OPS, SE
from model.utils import get_stride_for_cell_type


class Cell(nn.Module):
    _COEF = 0.1

    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type

        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, affine=False, channel_mult=CHANNEL_MULT)
        self.use_se = use_se
        self._num_nodes = len(arch)
        _ops = []
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            _ops.append(op)
        self._ops = nn.Sequential(*_ops)

        # SE
        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        s = self._ops(s)
        s = self.se(s) if self.use_se else s
        return skip + Cell._COEF * s


class CellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0

        self.cell_type = 'ar_nn'

        # s0 will the random samples
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)

        self.use_mix_log_cdf = False
        if self.use_mix_log_cdf:
            self.param = MixLogCDFParam(num_z, num_mix=3, num_ftr=self.conv.hidden_dim, mirror=mirror)
        else:
            # 0.1 helps bring mu closer to 0 initially
            self.mu = ARELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, masked=True, zero_diag=False,
                                weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z, ftr):
        s = self.conv(z, ftr)

        if self.use_mix_log_cdf:
            logit_pi, mu, log_s, log_a, b = self.param(s)
            new_z, log_det = mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b)
        else:
            mu = self.mu(s)
            new_z = (z - mu)
            log_det = torch.zeros_like(new_z)

        return new_z, log_det


class PairedCellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)

        log_det1 += log_det2
        return new_z, log_det1