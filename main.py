#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from glob import glob
import os

import tensorflow as tf
import numpy as np

from model import denoiser
from utils import *

tf.set_random_seed(0)
np.random.seed(0)




parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='# size of a patch')
parser.add_argument('--stride_size', dest='stride_size', type=int, default=128, help='# size of the stride')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--pile', dest='pile', type=int, default=1, help='size of the pile')  ## SERIES MULTI-TEMPORAL
parser.add_argument('--miso', action='store_true', help='if True, multi-input & single output')
parser.add_argument('--copy_input', action='store_true', help='if True, two noisy samples of the same image')
parser.add_argument('--load_all', action='store_true', help='if True, the data will be loaded in the load_train function, else at each batch')


parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default="./checkpoint",
                    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default="./sample", help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default="./test/", help='test sample are saved here')
parser.add_argument('--test_config', dest='test_config', default="default", help='how the testing subpiles are computed : default, reverse or all')


parser.add_argument('--eval_set', dest='eval_set', default='./data/evaluation/', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='./data/test/noisy/', help='dataset for testing')


args = parser.parse_args()

def denoiser_train(denoiser, lr):
    data = load_train_data(load_all=args.load_all)
    eval_data, eval_files = load_sar_images(args.eval_set, args.pile)  
    denoiser.train(data, eval_data, eval_files, eval_set=args.eval_set, batch_size=args.batch_size,
                   ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, step=0, pat_size=args.patch_size, stride=args.stride_size, eval_every_epoch=2)
    

def denoiser_test(denoiser):
    denoiser.test(test_set=args.test_set, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir, pile=args.pile, mode=args.test_config)




def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    #lr[10:20] = lr[0] / 10.0
    #lr[20:] = lr[0]/100
    
    if args.use_gpu:
        print("GPU\n")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            model = denoiser(sess, stride=args.stride_size, input_c_dim=args.pile, miso=args.miso, load_all=args.load_all)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, stride=args.stride_size, input_c_dim=args.pile, miso=args.miso, load_all=args.load_all)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
