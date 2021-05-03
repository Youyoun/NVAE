# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.multiprocessing import Process

import datasets
import utils
from configs import EncoderConfig, DecoderConfig, NVAEConfig
from src import arch_types
from src.model import AutoEncoder
from src.thirdparty.adamax import Adamax


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config(args):
    enc_config = EncoderConfig(args.num_channels_enc, args.num_preprocess_blocks, args.num_preprocess_cells,
                               args.num_cell_per_cond_enc)
    dec_config = DecoderConfig(args.num_channels_dec, args.num_postprocess_blocks, args.num_postprocess_cells,
                               args.num_cell_per_cond_dec)
    config = NVAEConfig(args.num_latent_scales, args.num_groups_per_scale, args.num_latent_per_group,
                        args.min_groups_per_scale, args.ada_groups, args.dataset, args.use_se, args.res_dist,
                        args.num_nf, args.num_x_bits, enc_config, dec_config)
    return config


class Main:
    def __init__(self, args):
        set_seed(args.seed)
        self.args = args

        self.logging = utils.Logger(args.global_rank, args.save)
        self.writer = utils.Writer(args.global_rank, args.save)

        # Get data loaders.
        self.train_queue, self.valid_queue, self.num_classes = datasets.get_loaders(args)
        args.num_total_iter = len(self.train_queue) * args.epochs
        self.warmup_iters = len(self.train_queue) * args.warmup_epochs

        self.arch_instance = arch_types.get_arch_cells(args.arch_instance)

        self.config = get_config(args)
        self.model = AutoEncoder(self.config, self.writer, self.arch_instance)
        self.model = self.model.cuda()

        self.logging.info('args = %s', args)
        self.logging.info('param size = %fM ', utils.count_parameters_in_M(self.model))
        self.logging.info('groups per scale: %s, total_groups: %d', self.model.groups_per_scale,
                          sum(self.model.groups_per_scale))

        self.cnn_optimizer, self.cnn_scheduler = self.init_optim()
        self.grad_scalar = GradScaler(2 ** 10)

        self.num_output = utils.num_output(args.dataset)
        self.bpd_coeff = 1. / np.log(2.) / self.num_output

        # if load
        self.checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
        self.global_step, self.init_epoch = 0, 0
        if self.args.cont_training:
            self.load_model()

    def init_optim(self):
        if self.args.fast_adamax:
            # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
            optimizer = Adamax(self.model.parameters(), self.args.learning_rate, weight_decay=self.args.weight_decay,
                               eps=1e-3)
        else:
            optimizer = torch.optim.Adamax(self.model.parameters(), self.args.learning_rate,
                                           weight_decay=self.args.weight_decay, eps=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               self.args.epochs - self.args.warmup_epochs - 1,
                                                               eta_min=self.args.learning_rate_min)
        return optimizer, scheduler

    def load_model(self):
        self.logging.info('loading the model.')
        checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
        self.init_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.cuda()
        self.cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        self.grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        self.cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['global_step']

    def run(self):
        for epoch in range(self.init_epoch, self.args.epochs):
            # update lrs.
            if args.distributed:
                self.train_queue.sampler.set_epoch(self.global_step + self.args.seed)
                self.valid_queue.sampler.set_epoch(0)

            if epoch > self.args.warmup_epochs:
                self.cnn_scheduler.step()

            # Logging.
            self.logging.info('epoch %d', epoch)

            # Training.
            train_nelbo = self.train_one_epoch()
            self.logging.info('train_nelbo %f', train_nelbo)
            self.writer.add_scalar('train/nelbo', train_nelbo, self.global_step)

            self.model.eval()
            # generate samples less frequently
            eval_freq = 1 if self.args.epochs <= 50 else 20
            if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
                with torch.no_grad():
                    num_samples = 16
                    n = int(np.floor(np.sqrt(num_samples)))
                    for t in [0.7, 0.8, 0.9, 1.0]:
                        logits = self.model.sample(num_samples, t)
                        output = self.model.decoder_output(logits)
                        output_img = output.mean if isinstance(output,
                                                               torch.distributions.bernoulli.Bernoulli) else output.sample(
                            t)
                        output_tiled = utils.tile_image(output_img, n)
                        self.writer.add_image('generated_%0.1f' % t, output_tiled, self.global_step)

                valid_neg_log_p, valid_nelbo = self.test(num_samples=10)
                self.logging.info('valid_nelbo %f', valid_nelbo)
                self.logging.info('valid neg log p %f', valid_neg_log_p)
                self.logging.info('valid bpd elbo %f', valid_nelbo * self.bpd_coeff)
                self.logging.info('valid bpd log p %f', valid_neg_log_p * self.bpd_coeff)
                self.writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
                self.writer.add_scalar('val/nelbo', valid_nelbo, epoch)
                self.writer.add_scalar('val/bpd_log_p', valid_neg_log_p * self.bpd_coeff, epoch)
                self.writer.add_scalar('val/bpd_elbo', valid_nelbo * self.bpd_coeff, epoch)

            save_freq = int(np.ceil(args.epochs / 100))
            if epoch % save_freq == 0 or epoch == (args.epochs - 1):
                if args.global_rank == 0:
                    self.logging.info('saving the model.')
                    torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                'optimizer': self.cnn_optimizer.state_dict(), 'global_step': self.global_step,
                                'args': self.args, 'arch_instance': self.arch_instance,
                                'scheduler': self.cnn_scheduler.state_dict(),
                                'grad_scalar': self.grad_scalar.state_dict()}, self.checkpoint_file)

        # Final validation
        valid_neg_log_p, valid_nelbo = self.test(num_samples=1000)
        self.logging.info('final valid nelbo %f', valid_nelbo)
        self.logging.info('final valid neg log p %f', valid_neg_log_p)
        self.writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
        self.writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
        self.writer.add_scalar('val/bpd_log_p', valid_neg_log_p * self.bpd_coeff, epoch + 1)
        self.writer.add_scalar('val/bpd_elbo', valid_nelbo * self.bpd_coeff, epoch + 1)
        self.writer.close()

    def train_one_epoch(self):
        alpha_i = utils.kl_balancer_coeff(num_scales=self.model.config.n_latent_scales,
                                          groups_per_scale=self.model.groups_per_scale, fun='square')
        nelbo = utils.AvgrageMeter()
        self.model.train()
        for step, x in enumerate(self.train_queue):
            x = x[0] if len(x) > 1 else x
            x = x.cuda()

            # change bit length
            x = utils.pre_process(x, args.num_x_bits)

            # warm-up lr
            if self.global_step < self.warmup_iters:
                lr = args.learning_rate * float(self.global_step) / self.warmup_iters
                for param_group in self.cnn_optimizer.param_groups:
                    param_group['lr'] = lr

            # sync parameters, it may not be necessary
            if step % 100 == 0:
                utils.average_params(self.model.parameters(), args.distributed)

            self.cnn_optimizer.zero_grad()
            with autocast():
                logits, log_q, log_p, kl_all, kl_diag = self.model(x)

                output = self.model.decoder_output(logits)
                kl_coeff = utils.kl_coeff(self.global_step, args.kl_anneal_portion * args.num_total_iter,
                                          args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

                recon_loss = utils.reconstruction_loss(output, x, crop=self.model.crop_output)
                balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

                nelbo_batch = recon_loss + balanced_kl
                loss = torch.mean(nelbo_batch)
                norm_loss = self.model.spectral_norm_parallel()
                bn_loss = self.model.batchnorm_loss()
                # get spectral regularization coefficient (lambda)
                if self.args.weight_decay_norm_anneal:
                    assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                    wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(
                        args.weight_decay_norm)
                    wdn_coeff = np.exp(wdn_coeff)
                else:
                    wdn_coeff = args.weight_decay_norm

                loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

            self.grad_scalar.scale(loss).backward()
            utils.average_gradients(self.model.parameters(), args.distributed)
            self.grad_scalar.step(self.cnn_optimizer)
            self.grad_scalar.update()
            nelbo.update(loss.data, 1)

            if (self.global_step + 1) % 100 == 0:
                if (self.global_step + 1) % 1000 == 0:  # reduced frequency
                    n = int(np.floor(np.sqrt(x.size(0))))
                    x_img = x[:n * n]
                    output_img = output.mean if isinstance(output,
                                                           torch.distributions.bernoulli.Bernoulli) else output.sample()
                    output_img = output_img[:n * n]
                    x_tiled = utils.tile_image(x_img, n)
                    output_tiled = utils.tile_image(output_img, n)
                    in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                    self.writer.add_image('reconstruction', in_out_tiled, self.global_step)

                # norm
                self.writer.add_scalar('train/norm_loss', norm_loss, self.global_step)
                self.writer.add_scalar('train/bn_loss', bn_loss, self.global_step)
                self.writer.add_scalar('train/norm_coeff', wdn_coeff, self.global_step)

                utils.average_tensor(nelbo.avg, args.distributed)
                self.logging.info('train %d %f', self.global_step, nelbo.avg)
                self.writer.add_scalar('train/nelbo_avg', nelbo.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.cnn_optimizer.state_dict()[
                    'param_groups'][0]['lr'], self.global_step)
                self.writer.add_scalar('train/nelbo_iter', loss, self.global_step)
                self.writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), self.global_step)
                self.writer.add_scalar('train/recon_iter',
                                       torch.mean(utils.reconstruction_loss(output, x, crop=self.model.crop_output)),
                                       self.global_step)
                self.writer.add_scalar('kl_coeff/coeff', kl_coeff, self.global_step)
                total_active = 0
                for i, kl_diag_i in enumerate(kl_diag):
                    utils.average_tensor(kl_diag_i, args.distributed)
                    num_active = torch.sum(kl_diag_i > 0.1).detach()
                    total_active += num_active

                    # kl_ceoff
                    self.writer.add_scalar('kl/active_%d' % i, num_active, self.global_step)
                    self.writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], self.global_step)
                    self.writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], self.global_step)
                self.writer.add_scalar('kl/total_active', total_active, self.global_step)

            self.global_step += 1

        utils.average_tensor(nelbo.avg, args.distributed)
        return nelbo.avg

    def test(self, num_samples):
        if args.distributed:
            dist.barrier()
        nelbo_avg = utils.AvgrageMeter()
        neg_log_p_avg = utils.AvgrageMeter()
        self.model.eval()
        for step, x in enumerate(self.valid_queue):
            x = x[0] if len(x) > 1 else x
            x = x.cuda()

            # change bit length
            x = utils.pre_process(x, args.num_x_bits)

            with torch.no_grad():
                nelbo, log_iw = [], []
                for k in range(num_samples):
                    logits, log_q, log_p, kl_all, _ = self.model(x)
                    output = self.model.decoder_output(logits)
                    recon_loss = utils.reconstruction_loss(output, x, crop=self.model.crop_output)
                    balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                    nelbo_batch = recon_loss + balanced_kl
                    nelbo.append(nelbo_batch)
                    log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=self.model.crop_output))

                nelbo = torch.mean(torch.stack(nelbo, dim=1))
                log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

            nelbo_avg.update(nelbo.data, x.size(0))
            neg_log_p_avg.update(- log_p.data, x.size(0))

        utils.average_tensor(nelbo_avg.avg, args.distributed)
        utils.average_tensor(neg_log_p_avg.avg, args.distributed)
        if args.distributed:
            # block to sync
            dist.barrier()
        self.logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
        return neg_log_p_avg.avg, nelbo_avg.avg


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn = fn(args)
    fn.run()
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, Main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, Main, args)
