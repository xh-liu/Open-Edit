import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle
import pdb

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:10002')
        parser.add_argument('--num_gpu', type=int, default=8, help='num of gpus for cluter training')
        parser.add_argument('--name', type=str, default='open-edit', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--vocab_path', type=str, default='vocab/conceptual_vocab.pkl', help='path to vocabulary') 
        parser.add_argument('--vse_enc_path', type=str, default='checkpoints/conceptual_model_best.pth.tar', help='path to the pretrained text encoder')
        parser.add_argument('--edge_model_path', type=str, default='checkpoints/bdcn_pretrained_on_bsds500.pth', help='path to the pretrained edge extractor')
        parser.add_argument('--model', type=str, default='OpenEdit', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralsync_batch', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        parser.add_argument('--img_size', type=int, default=224, help='image size')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/conceptual/')
        parser.add_argument('--dataset_mode', type=str, default='conceptual')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='openedit', help='generator model')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # for encoder
        parser.add_argument('--netE', type=str, default='resnetbdcn')
        parser.add_argument('--edge_nc', type=int, default=1)
        parser.add_argument('--edge_level', type=int, default=41)
        parser.add_argument('--edge_tanh', action='store_true')

        # for image-specific finetuning
        parser.add_argument('--reg_weight', type=float, default=1e-4)
        parser.add_argument('--perturbation', action='store_true')
        parser.add_argument('--manipulation', action='store_true')
        parser.add_argument('--img_path', type=str)
        parser.add_argument('--ori_cap', type=str)
        parser.add_argument('--new_cap', type=str)
        parser.add_argument('--global_edit', action='store_true')
        parser.add_argument('--alpha', type=int, default=5)
        parser.add_argument('--optimize_iter', type=int, default=50)

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain and save:
            self.save_options(opt)

        self.opt = opt
        return self.opt
