# Copyright 2019 Intel Corporation.

import logging
import six

import numpy as np
import plaidml2.edsl as edsl
from plaidml2.ffi import ForeignObject, decode_str, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    ffi_call(lib.plaidml_op_init)


ffi.init_once(__init, 'plaidml_op_init')


def op(op_name, args):
    value = edsl.Value(args)
    return edsl.Value(ffi_call(lib.plaidml_op_make, op_name.encode(), value.as_ptr()))


def abs(x):
    return op('abs', [x]).as_tensor()


def argmax(x, axis=-1):
    return op('argmax', [x, axis]).as_tensor()


def binary_crossentropy(targets, preds, epsilon):
    return op('binary_crossentropy', [targets, preds, epsilon]).as_tensor()


def clip(x, min=None, max=None):
    return op('clip', [x, min, max]).as_tensor()


def concatenate(tensors, axis=-1):
    return op('concatenate', [tensors, axis]).as_tensor()


def convolution(inputs, filters, strides, dilations, data_dilations, filter_shape, groups,
                autopad_mode, manual_padding, input_layout, filter_layout, group_layout,
                winograd_allowed, name, autogroup_mode, deriv_mode, result_shape):
    if isinstance(strides, np.ndarray):
        strides = strides.to_list()
    if isinstance(dilations, np.ndarray):
        dilations = dilations.to_list()
    if isinstance(data_dilations, np.ndarray):
        data_dilations = data_dilations.to_list()
    if isinstance(filter_shape, np.ndarray):
        filter_shape = filter_shape.to_list()
    if isinstance(manual_padding, np.ndarray):
        manual_padding = manual_padding.to_list()
    if isinstance(result_shape, np.ndarray):
        manual_padding = manual_padding.to_list()
    return op("convolution", [
        inputs, filters, strides, dilations, data_dilations, filter_shape, groups, autopad_mode,
        manual_padding, input_layout, filter_layout, group_layout, winograd_allowed, name,
        autogroup_mode, deriv_mode, result_shape
    ]).as_tensor()


def cumprod(x, axis):
    return op('cumprod', [x, axis]).as_tensor()


def cumsum(x, axis):
    return op('cumsum', [x, axis]).as_tensor()


def dot(x, y):
    return op('dot', [x, y]).as_tensor()


def elu(x, alpha=1.0):
    return op('elu', [x, alpha]).as_tensor()


def expand_dims(x, axis=-1):
    return op('expand_dims', [x, axis]).as_tensor()


def flip(x, axis=None):
    return op('flip', [x, axis]).as_tensor()


def hard_sigmoid(x, slope):
    return op('hard_sigmoid', [x, slope]).as_tensor()


def max(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('max', [x, axis, keepdims]).as_tensor()


def mean(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('mean', [x, axis, keepdims]).as_tensor()


def min(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('min', [x, axis, keepdims]).as_tensor()


def relu(x, alpha=None, max_value=None, threshold=0.):
    return op('relu', [x, alpha, max_value, threshold]).as_tensor()


def repeat(x, repeats, axis):
    return op('repeat', [x, repeats, axis]).as_tensor()


def sigmoid(x):
    return op('sigmoid', [x]).as_tensor()


def softmax(x, axis=None):
    return op('softmax', [x, axis]).as_tensor()


def prod(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('prod', [x, axis, keepdims]).as_tensor()


def pool(x, pool_mode, pool_size, strides, autopadding, manual_padding, data_layout, use_ceil,
         include_pad_in_avg):
    if isinstance(pool_size, np.ndarray):
        pool_size = pool_size.tolist()
    if isinstance(strides, np.ndarray):
        strides = strides.tolist()
    if isinstance(manual_padding, np.ndarray):
        manual_padding = manual_padding.to_list()
    return op("pool", [
        x, pool_mode, pool_size, strides, autopadding, manual_padding, data_layout, use_ceil,
        include_pad_in_avg
    ]).as_tensor()


def slice_of(x, slices):
    # Note: ellipses and too-short slice lists must be handled by the calling op if desired
    reformatted_slices = list()
    for s in slices:
        if isinstance(s, six.integer_types):
            reformatted_slices.append(s)
            continue
        if isinstance(s, slice):
            reformatted_slices.append([s.start, s.stop, s.step])
            continue
        raise ValueError("Unexpected type {} used to slice tensor".format(type(s)))
    return op("slice", [x, reformatted_slices]).as_tensor()


def spatial_padding(x, lo_pads, hi_pads, data_layout):
    if isinstance(lo_pads, np.ndarray):
        lo_pads = lo_pads.tolist()
    if isinstance(hi_pads, np.ndarray):
        hi_pads = hi_pads.to_list()
    return op("spatial_padding", [x, lo_pads, hi_pads, data_layout]).as_tensor()


def square(x):
    return x * x


def squeeze(x, axis):
    return op("squeeze", [x, axis]).as_tensor()


def sum(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('sum', [x, axis, keepdims]).as_tensor()


def tile(x, n):
    if isinstance(n, np.ndarray):
        n = n.tolist()
    return op('tile', [x, n]).as_tensor()


def transpose(x, pattern=None):
    if isinstance(pattern, np.ndarray):
        pattern = pattern.tolist()
    return op('transpose', [x, pattern]).as_tensor()


def variance(x, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('variance', [x, axis, keepdims]).as_tensor()
