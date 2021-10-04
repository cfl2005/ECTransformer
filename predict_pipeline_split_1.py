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
from pynvml import *

import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from train import translate, translate_encode, translate_decode, translate_decode_split
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp
#from torchvision import datasets, transforms
from torch.multiprocessing import Pool, Manager, Queue

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model_split import *


import warnings
warnings.filterwarnings('ignore')
nvmlInit() #初始化

# 文本转模型输入
def get_sample(sent):
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    #batch_input = torch.LongTensor(np.array(src_tokens)).to(torch.device('cpu')) #config.device)
    return src_tokens

# 把文本处理后发送到队列中
def send_dat(sentences, q_text, q_result):
    """
    发送数据
    """
    print('sending text...')
    total = len(sentences)
    for txt in sentences:
        batch_input = get_sample(txt)
        q_text.put(batch_input)
    
    i = 0 
    print('receive result...')
    while 1:
        ret = q_result.get()
        if type(ret)==str:
            pass
            print('result:', ret)
        else:
            pass
            #print('result:', type(ret))
        i+=1
        if i>=total:
            break;
    print('task end...')
    

def proc_encode(q_text, q_enc, model):
    """
    编码器
    """
    while True:
        #print("proc_encode.")
        dat = q_text.get()
        # print('q_text dat:',type(dat))
        batch_input = torch.LongTensor(np.array(dat)).to(config.device)
        src_enc, src_mask = translate_encode(batch_input, model)
        # print('src_enc:',type(src_enc))
        q_enc.put( (src_enc,src_mask) )

def proc_decode(q_enc, q_result, model_decode, model_generator):
    """
    解码器
    """
    while True:
        # 接收数据
        #print("proc_decode rev:")
        dat, src_mask = q_enc.get()
        #if not dat is None:
        src_enc = dat.clone()
        del dat
        #print('src_enc:',type(src_enc))
        #translation = translate_decode(src_enc, src_mask, model, use_beam=True)
        translation = translate_decode_split(src_enc, src_mask, model_decode, model_generator, use_beam=True)
        #print('translation:', translation)
        q_result.put(translation)

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

    logfile = 'report.txt'

    # 开始计时
    start = time.time()
    print(' NMT Task: Pipeline '.center(40,'-'))
    print('total sentences:%d'%total_sent)
    print('Building model...')

    # 创建拆分后的模型
    model_encoder, model_decoder, model_generator = make_split_model(
                        config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_encoder.share_memory()
    model_decoder.share_memory()
    model_generator.share_memory()

   
    # 加载模型

    print('Loading model...')
    model_encoder.load_state_dict(torch.load(config.model_path_encoder))
    model_decoder.load_state_dict(torch.load(config.model_path_decoder))
    model_generator.load_state_dict(torch.load(config.model_path_generator))
    
    model_encoder.eval()
    model_decoder.eval()
    model_generator.eval()
    
    print('create Queue...')
    q_text = Queue()
    q_enc = Queue()
    q_result = Queue()

    print('create process...')
    p_encode = mp.Process(target=proc_encode, args=(q_text, q_enc, model_encoder))
    p_decode = mp.Process(target=proc_decode, args=(q_enc, q_result, model_decoder, model_generator))

    print('process start...')
    p_decode.start()
    p_encode.start()

    loadstime = (time.time() - start)*1000

    # 开始计时
    start = time.time()
    print('Create sender process...')
    p_sender = mp.Process(target=send_dat, args=(sentences, q_text, q_result))
    p_sender.start()

    print('process join...')
    #p_encode.join()
    #p_decode.join()
    memory = GPU_memory(0)
    p_sender.join()

    predict_time = (time.time() - start)*1000
    avetime = predict_time/total_sent

    print('加载模型用时:%f 毫秒' % loadstime )
    print('Used Memory:%d MB'%memory)
    print('预测总计用时:%f 毫秒' % predict_time )
    print('预测单句用时:%f 毫秒' % avetime )

    result = 'Pipeline', total_sent, memory, loadstime, predict_time, avetime
    # 追加到日志文件
    savetofile(result, logfile)

    p_encode.terminate()
    p_decode.terminate()

