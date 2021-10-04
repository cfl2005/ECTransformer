#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import re
import json
import time
import utils
import config
import logging
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import *
from translate import GPU_memory
from pynvml import *

import warnings
warnings.filterwarnings('ignore')

from Pytorch_Memory_Utils.gpu_mem_track import MemTracker  # 引用显存跟踪代码
gpu_tracker = MemTracker()      # 创建显存检测对象

# -----------------------------------------

# 测试单元
def test_load_model(mid=0):
    print(' Unit Test: Load Model '.center(40,'-'))
    # 
    nvmlInit() #初始化
    if mid==0:
        # 加载模型
        gpu_tracker.track()                  # 开始检测
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        print('Loading model: Transformer...')
        model.load_state_dict(torch.load(config.model_path))
        time.sleep(1)

        gpu_tracker.track()                  # 开始检测
        memory0 = GPU_memory(0)
        print('Used Memory:%d MB' % memory0)
        del model
        time.sleep(1)

    # -----------------------------------------
    if mid==1:
        # 创建模型
        gpu_tracker.track()                  # 开始检测
        model_encoder = make_model_encode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)


        # 加载模型
        print('Loading model:encoder...')
        model_encoder.load_state_dict(torch.load(config.model_path_encoder))
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测

        memory1 = GPU_memory(0)
        print('Used Memory:%d MB' % memory1)
        del model_encoder
        time.sleep(1)

    # -----------------------------------------
    if mid==2:
        # 创建并加载模型
        gpu_tracker.track()                  # 开始检测
        model_decoder, model_generator = make_model_decode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        print('Loading model:decoder,generator ...')
        model_decoder.load_state_dict(torch.load(config.model_path_decoder))
        model_generator.load_state_dict(torch.load(config.model_path_generator))
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测

        memory2 = GPU_memory(0)
        print('Used Memory:%d MB' % memory2)
        time.sleep(1)
        del model_decoder
        del model_generator

# 加载空模型，只定义看有没有消耗显存
def test_load_blank_model(mid=0):
    print(' Unit Test: Load Blank Model '.center(40,'-'))
    # 
    nvmlInit() #初始化
    if mid==0:
        # 加载模型
        gpu_tracker.track()                  # 开始检测
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        print('Loading model: Transformer...')
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测

        memory0 = GPU_memory(0)
        print('Used Memory:%d MB' % memory0)
        del model
        time.sleep(1)

    # -----------------------------------------
    if mid==1:
        # 创建模型
        gpu_tracker.track()                  # 开始检测
        model_encoder = make_model_encode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)


        # 加载模型
        print('Loading model:encoder...')
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测

        memory1 = GPU_memory(0)
        print('Used Memory:%d MB' % memory1)
        del model_encoder
        time.sleep(1)

    # -----------------------------------------
    if mid==2:
        # 创建并加载模型
        gpu_tracker.track()                  # 开始检测
        model_decoder, model_generator = make_model_decode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        print('Loading model:decoder,generator ...')
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测

        memory2 = GPU_memory(0)
        print('Used Memory:%d MB' % memory2)
        time.sleep(1)
        del model_decoder
        del model_generator


# 加载模型：定义及权重同时加载
def test_load_whole_model(mid=0):
    print(' Unit Test: Load Whole Model '.center(40,'-'))
    # 
    outpath = './experiment/'
    model_path = os.path.join(outpath, 'model_all.pth')
    encode_path = os.path.join(outpath, 'encoder_all.pth')
    decoder_path = os.path.join(outpath, 'decoder_all.pth')
    generator_path = os.path.join(outpath, 'generator_all.pth')

    nvmlInit() #初始化
    if mid==0:
        # 加载模型
        gpu_tracker.track()                  # 开始检测
        print('Loading model: Transformer...')
        model = torch.load(model_path)
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测
        memory0 = GPU_memory(0)
        print('Used Memory:%d MB' % memory0)
        time.sleep(1)
        del model

    # -----------------------------------------
    if mid==1:
        # 加载模型
        gpu_tracker.track()                  # 开始检测
        print('Loading model:encoder...')
        model_encoder = torch.load(encode_path)
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测
        memory1 = GPU_memory(0)
        print('Used Memory:%d MB' % memory1)
        time.sleep(1)
        del model_encoder

    # -----------------------------------------
    if mid==2:
        # 加载模型
        gpu_tracker.track()                  # 开始检测
        print('Loading model:decoder,generator ...')
        model_decoder = torch.load(decoder_path)
        model_generator = torch.load(generator_path)
        time.sleep(1)
        gpu_tracker.track()                  # 开始检测
        memory2 = GPU_memory(0)
        print('Used Memory:%d MB' % memory2)
        time.sleep(1)
        del model_decoder
        del model_generator



if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser(description='NMT Model Load Test')
    parser.add_argument('--mtype', type=int, default=1, help='0=load model weight,1=whole model, 2=blank model')
    parser.add_argument('--mid', type=int, default=0, help='0=full model 1=encode model 2=decode model ')
    args = parser.parse_args()
    mid = args.mid
    mtype = args.mtype

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    if mtype==0:
        test_load_model(mid=mid)
    if mtype==1:
        test_load_whole_model(mid=mid)
    if mtype==2:
        test_load_blank_model(mid=mid)



