#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx
import numpy as np
import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_from_vgg16(ctx, symbol, vgg16_args, vgg16_aux):
    args = vgg16_args.copy()
    auxs = vgg16_aux.copy()
    for k, v in args.items():
        if(v.context != ctx):
            args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(args[k])
    for k, v in auxs.items():
        if (v.context != ctx):
            auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(auxs[k])
    data_shape = (1, 3, 513, 513)
    arg_names = symbol.list_arguments()
    arg_shapes, _, _ = symbol.infer_shape(data=data_shape)
    reset_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in \
                         zip(arg_names, arg_shapes) if x[0] in \
                        ["fc8_exper_bias", "fc8_exper_weight", \
                         "fc6_weight", "fc6_bias", \
                         "fc7_weight", "fc7_bias"]])
    init = mx.initializer.Normal(sigma=0.01)

    args.update(reset_params)
    for k, v in args.items():
        if k in ["fc8_exper_weight", "fc6_weight", "fc7_weight"]:
            init(k, args[k])
    for k in ["fc8_bias", "fc8_weight"]:
        print args.pop(k)

    return args
