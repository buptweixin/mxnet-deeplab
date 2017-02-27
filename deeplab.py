#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx
from data import FileIter
from solver import Solver
import init_deeplab
import symbol_deeplab
import sys, os
import argparse
import numpy as np
import logging
import cv2


def main():
    epoch = 74
    prefix = "VGG_FC_ILSVRC_16_layers"
    pwd = os.getcwd()
    os.chdir("../")
    _, args, auxs = mx.model.load_checkpoint(prefix, epoch)
    os.chdir(pwd)

    deeplabv2_symbol_train = symbol_deeplab.get_deeplab_symbol_train()
    args = init_deeplab.init_from_vgg16(mx.gpu(), deeplabv2_symbol_train, args, auxs)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ctx = mx.gpu()
    train_dataiter = FileIter(
        root_dir    = parser_args.root,
        flist_name  = "train.txt",
        shrink      = 8,
        rgb_mean    = (123.68, 116.779, 103.939),
    )
    val_dataiter = FileIter(
        root_dir    = parser_args.root,
        flist_name  = "test.txt",
        shrink      = 8,
        rgb_mean    = (123.68, 116.779, 103.939),
    )
    if parser_args.retrain:
        begin_epoch = 50
    else:
        begin_epoch = 0
    model = Solver(
        ctx           = ctx,
        symbol        = deeplabv2_symbol_train,
        begin_epoch   = begin_epoch,
        num_epoch     = 50,
        arg_params    = args,
        aux_params   = auxs,
        learning_rate = 0.001,
        momentum      = 0.9,
        wd            = 0.0005
    )
    deeplab_model_prefix="DeepLab-V2"
    model.fit(
        train_data         = train_dataiter,
        eval_data          = val_dataiter,
        batch_end_callback = mx.callback.Speedometer(1, 10),
        epoch_end_callback = mx.callback.do_checkpoint(deeplab_model_prefix)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating deeplab model of MXNet")
    parser.add_argument('--root', type=str, help="The root dir of data and list file.")
    parser.add_argument('--retrain', action='store_true', default=False, help="true means continue training.")
    parser_args = parser.parse_args()
    logging.info(parser_args)
    main()
