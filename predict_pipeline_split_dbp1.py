#!/usr/bin/env python3
# coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import re
import json
import logging
from math import *
# from flask import Flask, request, render_template, jsonify, json
import time
import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from train import translate, translate_encode, translate_decode, translate_decode_split
from utils import english_tokenizer_load
from model import make_split_model
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager, Queue
from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model_split import *

import warnings
warnings.filterwarnings('ignore')

nvmlInit()  # 初始化


# 把文本处理后发送到队列中
def send_dat(sentences, q_text, q_result):
    """
    发送数据
    """
    print('sending text...')
    # 句子按batch_size拆分
    batch_size = 64
    total = len(sentences)
    sentlist = [sentences[i*batch_size:(i+1)*batch_size] for i in range(ceil(total/batch_size))]

    for txts in sentlist:
        batch_text = get_sample(txts)
        # print('batch_text:', type(batch_text))
        # print('batch_text:', batch_text)
        q_text.put(batch_text)
    
    i = 0 
    print('receive result...')
    while 1:
        ret = q_result.get()
        if type(ret) == list:
            pass
            #print('result:\n', '\n'.join(ret))
            i += len(ret)
        else:
            pass
            #print('result:', type(ret))
        #i+=1
        if i>=total:
            break;
    print('task end...')

def proc_encode(q_text, q_enc, model):
    """
    编码器
    """
    while True:
        # print("proc_encode.")
        dat = q_text.get()
        # print('q_text dat:',type(dat))  #np.array(dat)
        batch_input = torch.LongTensor(dat).to(config.device) 
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
        src_enc = dat.clone()
        del dat
        #print('src_enc:',type(src_enc))
        translation = translate_decode_split(src_enc, src_mask, model_decode, model_generator, use_beam=True)
        #print('translation:', translation)
        q_result.put(translation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model Double Process Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--logfile', type=str, default="report.txt", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    logfile = args.logfile

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

    # 开始计时
    start = time.time()
    print(' NMT Task: Dbl Pipeline '.center(40, '-'))
    print('total sentences:%d' % total_sent)
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
    # 流水线1队列
    q_text1 = mp.Queue()
    q_enc1 = mp.Queue()
    q_result1 = mp.Queue()

    # 流水线2队列
    q_text2 = mp.Queue()
    q_enc2 = mp.Queue()
    q_result2 = mp.Queue()

    print('create process...')
    # 流水线1进程
    p_encode1 = mp.Process(target=proc_encode, args=(q_text1, q_enc1, model_encoder))
    p_decode1 = mp.Process(target=proc_decode, args=(q_enc1, q_result1, model_decoder, model_generator))

    # 流水线2进程
    p_encode2 = mp.Process(target=proc_encode, args=(q_text2, q_enc2, model_encoder))
    p_decode2 = mp.Process(target=proc_decode, args=(q_enc2, q_result2, model_decoder, model_generator))

    print('process start...')
    p_encode1.start()
    p_encode2.start()

    print('decoder start....')
    p_decode1.start()
    p_decode2.start()

    loadstime = (time.time() - start) * 1000

    # 开始计时
    start = time.time()
    print('Create sender process...')
    p_sender1 = mp.Process(target=send_dat, args=(sentences, q_text1, q_result1))
    p_sender2 = mp.Process(target=send_dat, args=(sentences, q_text2, q_result2))

    p_sender1.start()
    p_sender2.start()

    print('process join...')
    memory = GPU_memory(0)
    p_sender1.join()
    p_sender2.join()

    predict_time = (time.time() - start) * 1000
    avetime = predict_time / (total_sent) #/2

    print('加载模型用时:%f 毫秒' % loadstime)
    print('Used Memory:%d MB' % memory)
    print('预测总计用时:%f 毫秒' % predict_time)
    print('预测单句用时:%f 毫秒' % avetime)

    result = 'DBLPipeline', total_sent, memory, loadstime, predict_time, avetime
    # 追加到日志文件
    savetofile(result, logfile)

    p_encode1.terminate()
    p_decode1.terminate()
    p_encode2.terminate()
    p_decode2.terminate()
