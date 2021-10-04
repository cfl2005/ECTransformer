#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import argparse
import os
import re
import json
import logging
#from flask import Flask, request, render_template, jsonify, json


#import multiprocessing
import time

import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from train import translate, translate_encode, translate_decode
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp
#from torchvision import datasets, transforms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')

# 文本转模型输入
def get_sample(sent):
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    return batch_input

def translate_sample(txt, model, beam_search=True):
    print('text:', txt)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
    batch_input = torch.LongTensor(np.array(sample)).to(config.device)
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)


def translate_sample1(txts, model, beam_search=True):
    res = {}
    print('texts:', txts)
    batch_input = []

    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3

    for txt in txts.splitlines():
        sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
        batch_input.append(torch.LongTensor(np.array(sample)).to(config.device))

    #batch_input = get_sample(txt)
    #batch_input = 
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)

def proc_encode(pipe, texts, model):
    """
    发送数据
    """
    #while True:
    for i,txt in enumerate(texts.splitlines()):
        batch_input = get_sample(txt)
        src_enc = translate_encode(batch_input, model)
        print('src_enc:',type(src_enc))
        #dat = src_enc.cpu().numpy()
        #print('dat:',type(dat))
        print("send dat: %s" %i)
        #发送数据
        pipe.send(src_enc)

def proc_decode(pipe, model):
    """
    接收数据
    """
    while True:
        # 接收数据
        print("proc_decode rev:")
        src_enc = pipe.recv()
        #print('dat:',type(dat))
        #src_enc = torch.from_numpy(dat)
        print('src_enc:',type(src_enc))
        translation = translate_decode(src_enc, model, use_beam=True)
        print('translation:', translation)

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=1, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')


    args = parser.parse_args()

    # 开始计时
    start = time.time()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')
    #mp.set_start_method('forkserver')

    # 加载模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
   
    processes = []
    texts = '''Moreover, this war has been funded differently from any other war in America’s history –perhaps in any country’s recent history.
    But China’s policy failures should come as no surprise.
    The results have been devastating.
    Wind and solar are often presented as alternatives to oil and gas, but they cannot compete with traditional sources for electricity generation.
    Nor are Muslim women alone.'''
    
    
    print('create process...')

    #返回两个连接对象
    pipe2,pipe1 = mp.Pipe(duplex=False)
    p_encode = mp.Process(target=proc_encode, args=(pipe1, texts, model))
    p_decode = mp.Process(target=proc_decode, args=(pipe2, model))

    print('process start...')
    p_encode.start()
    p_decode.start()

    print('process join...')
    p_encode.join()
    p_decode.join()

    estime = (time.time() - start)*1000
    print('用时:%f 毫秒' % estime )
    
