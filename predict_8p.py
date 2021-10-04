#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import argparse
import os
import re
import json
import logging
from math import *
#from flask import Flask, request, render_template, jsonify, json
import sys
import time
import utils
import config
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from train import translate
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp
from translate import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')

nvmlInit() #初始化

def translate_batch_sub(sentences, model=None, batch_size=64, show_result=0):
    pass
    # 加载模型
    if model is None:
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        # model.share_memory()
        print('Loading model...')
        model.load_state_dict(torch.load(config.model_path))
        # model.eval()

    result = translate_batch(sentences, model, batch_size=batch_size)
    if show_result:
        print('result:\n', '\n'.join(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model Double Process Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--show_result', type=int, default=0, help='show result')
    parser.add_argument('--logfile', type=str, default="report.txt", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    logfile = args.logfile
    batch_size = args.batch_size
    show_result = args.show_result

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    mp.set_start_method('spawn')

    # 加载数据文件
    txts = readtxt(datafile)
    if txts:
        sentences = txts.splitlines()
        sentences = list(filter(None, sentences))
        total_sent = len(sentences)
    else:
        print('data file error')
        sys.exit()

    # 数据拆分成两个
    # pos = total_sent // 2
    # split_sents = [sentences[:pos], sentences[pos:]]
    
    # 启动显存监测记录
    obj = GPU_MEM()
    obj.build()
    obj.start()

    # 开始计时
    start = time.time()
    print(' NMT Task: 8Process Batch '.center(40,'-'))
    print('total sentences:%d'%total_sent)
    print('Building model...')

    # 加载模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    loadstime = (time.time() - start)*1000

    # 开始计时
    start = time.time()
    print('Create process...')
    # 主进程加载模型及权重，共享给子进程
    p1 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p2 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p3 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p4 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p5 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p6 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p7 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    p8 = mp.Process(target=translate_batch_sub, args=(sentences, model, batch_size, show_result))
    
    # 在子进程中加载模型定义及权重
    #p1 = mp.Process(target=translate_batch_sub, args=(sentences, ))
    #p2 = mp.Process(target=translate_batch_sub, args=(sentences, ))
    
    print('Start process...')
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    print('Process running...')
    memory = GPU_memory(0)
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    # 停止显存监测
    obj.stop()
    # print('mem data:', obj.data)
    print('mem_ave:', obj.mem_ave())
    print('mem_max:', obj.mem_max())

    predict_time = (time.time() - start)*1000
    avetime = predict_time/total_sent
    
    print('加载模型用时:%f 毫秒' % loadstime )
    print('Used Memory:%d MB'%memory)
    print('预测总计用时:%f 毫秒' % predict_time )
    print('预测单句用时:%f 毫秒' % avetime )

    result = {'name':'8Process', 'total_sent':total_sent,
                'memory':memory, 'loadstime': loadstime, 
                'predict_time':predict_time, 'avetime':avetime, 'batch_size':batch_size,
                'mem data':obj.data, 'mem_ave': obj.mem_ave(), 'mem_max':obj.mem_max()
             }

    # 追加到日志文件
    savetofile(json.dumps(result), logfile)

