#!/usr/bin/env python3
#coding:utf-8

__author__ = 'caifenglong'


import argparse
import os
import re
import json
import logging
#from flask import Flask, request, render_template, jsonify, json

#import multiprocessing
import time
from pynvml import *

import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from train import translate
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp
#from torchvision import datasets, transforms

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')

nvmlInit() #初始化


# 文本转模型输入
def get_sample(sent):
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    return batch_input

# 单句子翻译
def translate_sample(txt, model, beam_search=True):
    #print('text:', txt)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
    batch_input = torch.LongTensor(np.array(sample)).to(config.device)
    ret = translate(batch_input, model, use_beam=beam_search)
    #print('translate:', ret)
    return ret

# 多句子翻译 
def translate_texts(sentences, model, beam_search=True):
    result = []
    for txt in sentences:
        if txt:
            ret = translate_sample(txt, model, beam_search=beam_search)
            result.append(ret)
    return result
    
def translate_batch(txts, model, beam_search=True):
    #print('texts:', txts)
    samples = []

    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3

    for txt in txts.splitlines():
        sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
        samples.append(sample)

    arr_samples = np.array(samples)
    print('arr_samples:', arr_samples.shape)
    batch_input = torch.LongTensor(arr_samples).to(config.device)
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)


# 读入文件
def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''

# 保存文本信息到文件
def savetofile(txt, filename, encoding='utf-8', method='a+'):
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt)+ '\n')
        return 1
    except :
        return 0

def GPU_memory(gpuid=0):
    NUM_EXPAND = 1024 * 1024
    #获取GPU i的handle，后续通过handle来处理
    handle = nvmlDeviceGetHandleByIndex(gpuid)
    #通过handle获取GPU i的信息
    info = nvmlDeviceGetMemoryInfo(handle)

    #gpu_memory_total = info.total / NUM_EXPAND  #GPU i的总显存
    gpu_memory_used = info.used / NUM_EXPAND  #转为MB单位
    #print('Total Memory:%d MB,  Used Memory:%d MB'% (gpu_memory_total, gpu_memory_used))
    return  gpu_memory_used

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model Double Process Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    #parser.add_argument('--logfile', type=str, required=1, default="", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    #logfile = args.logfile

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

    #logfile = 'SingleProcess_%d.txt'%total_sent
    logfile = 'report.txt'

    # 数据拆分成两个
    pos = total_sent // 2
    split_sents = [sentences[:pos], sentences[pos:]]
    
    # 开始计时
    start = time.time()
    print(' NMT Task: Double Process '.center(40,'-'))
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
    print('Create process...')
    # 创建两个进程进行预测
    p1 = mp.Process(target=translate_batch, args=(txts, model))
    # p1 = mp.Process(target=translate_texts, args=(sentences, model))
    # p2 = mp.Process(target=translate_texts, args=(sentences, model))
    # p3 = mp.Process(target=translate_texts, args=(sentences, model))
    # p4 = mp.Process(target=translate_texts, args=(sentences, model))

    start = time.time()
    print('Start process...')
    p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    print('Process running...')
    memory = GPU_memory(0)
    p1.join()
    # p2.join()
    # p3.join()
    # p4.join()


    predict_time = (time.time() - start)*1000
    avetime = predict_time/total_sent
    
    print('加载模型用时:%f 毫秒' % loadstime )
    print('Used Memory:%d MB'%memory)
    print('预测总计用时:%f 毫秒' % predict_time )
    # print('预测单句用时:%f 毫秒' % avetime )

    result = 'DoubleProcess', total_sent, memory, loadstime, predict_time, avetime
    # 追加到日志文件
    savetofile(result, logfile)


