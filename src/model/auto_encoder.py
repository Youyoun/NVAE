# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from configs import NVAEConfig, DataConfig
from distributions import Normal, DiscMixLogistic
from .cell import Cell, PairedCellAR, CHANNEL_MULT
from .neural_ar_operations import ARConv2d
from .neural_operations import EncCombinerCell, DecCombinerCell, Conv2D
from .utils import get_input_size, groups_per_scale
from src.thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish


class AutoEncoder(nn.Module):
    def __init__(self, model_config: NVAEConfig, dataset: str, writer, arch_instance):
        super(AutoEncoder, self).__init__()
        self.writer = writer
        self.arch_instance = arch_instance
        self.config = model_config
        self.dataset = dataset
        self.crop_output = self.dataset == "mnist"
        self.config.norm_flow
        self.groups_per_scale = groups_per_scale(self.config.latent.n_latent_scales,
                                                 self.config.latent.n_groups_per_scale,
                                                 self.config.latent.ada_groups,
                                                 minimum_groups=self.config.latent.min_groups_per_scale)

        self.vanilla_vae = self.config.latent.n_latent_scales == 1 and self.config.latent.n_groups_per_scale == 1

        # general cell parameters
        self.input_size = get_input_size(self.dataset)

        # used for generative purpose
        c_scaling = CHANNEL_MULT ** (self.config.encoder.n_preprocess_blocks + self.config.latent.n_latent_scales - 1)
        spatial_scaling = 2 ** (self.config.encoder.n_preprocess_blocks + self.config.latent.n_latent_scales - 1)
        prior_ftr0_size = (int(c_scaling * self.config.decoder.n_channels), self.input_size // spatial_scaling,
                           self.input_size // spatial_scaling)
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)
        self.z0_size = [self.config.latent.n_latent_per_group, self.input_size // spatial_scaling,
                        self.input_size // spatial_scaling]

        self.stem = self.init_stem()
        self.pre_process, mult = self.init_pre_process(mult=1)

        if self.vanilla_vae:
            self.enc_tower = []
        else:
            self.enc_tower, mult = self.init_encoder_tower(mult)

        self.with_nf = self.config.norm_flow.num_nf > 0
        self.num_flows = self.config.norm_flow.num_nf

        self.enc0 = self.init_encoder0(mult)
        self.enc_sampler, self.dec_sampler, self.nf_cells, self.enc_kv, self.dec_kv, self.query = \
            self.init_normal_sampler(mult)

        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = Conv2D(self.config.latent.n_latent_per_group, mult * self.config.encoder.n_channels,
                                       (1, 1),
                                       bias=True)
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)

        self.post_process, mult = self.init_post_process(mult)

        self.image_conditional = self.init_image_conditional(mult)

        # collect all norm params in Conv2D and gamma param in batchnorm
        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)

        print('len log norm:', len(self.all_log_norm))
        print('len bn:', len(self.all_bn_layers))
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def init_stem(self):
        Cout = self.config.encoder.n_channels
        Cin = 1 if self.dataset == 'mnist' else 3
        stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
        return stem

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.config.encoder.n_preprocess_blocks):
            for c in range(self.config.encoder.n_preprocess_cells):
                if c == self.config.encoder.n_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.config.encoder.n_channels * mult)
                    num_co = int(CHANNEL_MULT * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.config.use_se)
                    mult = CHANNEL_MULT * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.config.encoder.n_channels * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.config.use_se)

                pre_process.append(cell)

        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        for s in range(self.config.latent.n_latent_scales):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.config.encoder.n_cell_per_cond):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.config.encoder.n_channels * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.config.use_se)
                    enc_tower.append(cell)

                # add encoder combiner
                if not (s == self.config.latent.n_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.config.encoder.n_channels * mult)
                    num_cd = int(self.config.decoder.n_channels * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)

            # down cells after finishing a scale
            if s < self.config.latent.n_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.config.encoder.n_channels * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.config.use_se)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult

        return enc_tower, mult

    def init_encoder0(self, mult):
        num_c = int(self.config.encoder.n_channels * mult)
        cell = nn.Sequential(
            nn.ELU(),
            Conv2D(num_c, num_c, kernel_size=1, bias=True),
            nn.ELU())
        return cell

    def init_normal_sampler(self, mult):
        enc_sampler, dec_sampler, nf_cells = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        enc_kv, dec_kv, query = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.config.latent.n_latent_scales):
            for g in range(self.groups_per_scale[self.config.latent.n_latent_scales - s - 1]):
                # build mu, sigma generator for encoder
                num_c = int(self.config.encoder.n_channels * mult)
                cell = Conv2D(num_c, 2 * self.config.latent.n_latent_per_group, kernel_size=3, padding=1, bias=True)
                enc_sampler.append(cell)
                # build NF
                for n in range(self.num_flows):
                    arch = self.arch_instance['ar_nn']
                    num_c1 = int(self.config.encoder.n_channels * mult)
                    num_c2 = 8 * self.config.latent.n_latent_per_group  # use 8x features
                    nf_cells.append(PairedCellAR(self.config.latent.n_latent_per_group, num_c1, num_c2, arch))
                if not (s == 0 and g == 0):  # for the first group, we use a fixed standard Normal.
                    num_c = int(self.config.decoder.n_channels * mult)
                    cell = nn.Sequential(
                        nn.ELU(),
                        Conv2D(num_c, 2 * self.config.latent.n_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)

            mult = mult / CHANNEL_MULT

        return enc_sampler, dec_sampler, nf_cells, enc_kv, dec_kv, query

    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = nn.ModuleList()
        for s in range(self.config.latent.n_latent_scales):
            for g in range(self.groups_per_scale[self.config.latent.n_latent_scales - s - 1]):
                num_c = int(self.config.decoder.n_channels * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.config.decoder.n_cell_per_cond):
                        arch = self.arch_instance['normal_dec']
                        cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.config.use_se)
                        dec_tower.append(cell)

                cell = DecCombinerCell(num_c, self.config.latent.n_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)

            # down cells after finishing a scale
            if s < self.config.latent.n_latent_scales - 1:
                arch = self.arch_instance['up_dec']
                num_ci = int(self.config.decoder.n_channels * mult)
                num_co = int(num_ci / CHANNEL_MULT)
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch, use_se=self.config.use_se)
                dec_tower.append(cell)
                mult = mult / CHANNEL_MULT

        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.config.decoder.n_postprocess_blocks):
            for c in range(self.config.decoder.n_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.config.decoder.n_channels * mult)
                    num_co = int(num_ci / CHANNEL_MULT)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.config.use_se)
                    mult = mult / CHANNEL_MULT
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.config.decoder.n_channels * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.config.use_se)

                post_process.append(cell)

        return post_process, mult

    def init_image_conditional(self, mult):
        C_in = int(self.config.decoder.n_channels * mult)
        C_out = 1 if self.dataset == 'mnist' else 10 * self.config.decoder.n_mix_output
        return nn.Sequential(nn.ELU(),
                             Conv2D(C_in, C_out, 3, padding=1, bias=True))

    def forward(self, x):
        s = self.stem(2 * x - 1.0)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.enc0(s)  # this reduces the channel dimension
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)  # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        # apply normalizing flows
        nf_offset = 0
        for n in range(self.num_flows):
            z, log_det = self.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.num_flows
        all_q = [dist]
        all_log_q = [log_q_conv]

        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        # prior for z0
        dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]

        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.config.res_dist else Normal(mu_q,
                                                                                                          log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    # apply NF
                    for n in range(self.num_flows):
                        z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                        log_q_conv -= log_det
                    nf_offset += self.num_flows
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            if self.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        return logits, log_q, log_p, kl_all, kl_diag

    def sample(self, num_samples, t):
        scale_ind = 0
        z0_size = [num_samples] + self.z0_size
        dist = Normal(mu=torch.zeros(z0_size).cuda(), log_sigma=torch.zeros(z0_size).cuda(), temp=t)
        z, _ = dist.sample()

        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu, log_sigma, t)
                    z, _ = dist.sample()

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
                if cell.cell_type == 'up_dec':
                    scale_ind += 1

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)
        return logits

    def decoder_output(self, logits):
        if self.dataset == 'mnist':
            return Bernoulli(logits=logits)
        elif self.dataset in {'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq',
                              'lsun_bedroom_128', 'lsun_bedroom_256'}:
            return DiscMixLogistic(logits, self.config.decoder.n_mix_output, num_bits=self.config.norm_flow.n_bits)
        else:
            raise NotImplementedError

    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}  # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))

        return loss
