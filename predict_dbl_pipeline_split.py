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
            #print('result:', ret)
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
        dat = q_text.get(False)
        # print('q_text dat:',type(dat))
        if not dat is None:
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
        dat, src_mask = q_enc.get(False)
        if not dat is None:
            src_enc = dat.clone()
            del dat
            #print('src_enc:',type(src_enc))
            #translation = translate_decode(src_enc, src_mask, model, use_beam=True)
            translation = translate_decode_split(src_enc, src_mask, model_decode, model_generator, use_beam=True)
            #print('translation:', translation)
            q_result.put(translation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT Model Double Pipeline Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--pipelines', type=int, default=1, help='pipelines')
    args = parser.parse_args()
    datafile = args.datafile

    print(' NMT Task: DoublePipelineSplit '.center(40,'-'))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    mp.set_start_method('spawn')
    
    # pipeline个数
    num_pipelines = 2
    num_pipelines = args.pipelines

    # 加载数据文件
    print('Loading data file...')
    txts = readtxt(datafile)
    if txts:
        sentences = txts.splitlines()
        sentences = list(filter(None, sentences))
        total_sent = len(sentences)
    else:
        print('data file error')
        sys.exit()
    
    # 数据拆分成多块个
    split_sents = [sentences[i::num_pipelines] for i in range(num_pipelines)]
    #print(split_sents)
    #print('split_sents:', list(map(lambda x:len(x), split_sents)) )
    #sys.exit()

    logfile = 'report.txt'

    # 开始计时
    start = time.time()
    print('Total sentences:%d'%total_sent)

    pipelines = [{'Queue':[], 'Process':[],'Sender':[]} for i in range(num_pipelines)] 


    for i in range(num_pipelines):

        # 加载模型
        print('Building model...')
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        model.share_memory()
        print('Loading model...')
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

        encoder = model.encoder
        decoder = model.decoder
        src_embed = model.src_embed
        tgt_embed = model.tgt_embed
        model_generator = model.generator

        model_encode = Transformer_encode(encoder, src_embed)
        model_decode = Transformer_decode(decoder, tgt_embed)

        print('create Queue...')
        q_text = Queue()
        q_enc = Queue()
        q_result = Queue()
        pipelines[i]['Queue'].append(q_text)
        pipelines[i]['Queue'].append(q_enc)
        #pipelines[i]['Queue'].append(q_result)
        
        print('create process...')
        p_encode = mp.Process(target=proc_encode, args=(q_text, q_enc, model_encode))
        p_decode = mp.Process(target=proc_decode, args=(q_enc, q_result, model_decode, model_generator))
        pipelines[i]['Process'].append(p_decode)
        pipelines[i]['Process'].append(p_encode)

        '''
        print('process start...')
        p_decode.start()
        p_encode.start()
        '''

        '''
        print('Create sender process...')
        p_sender = mp.Process(target=send_dat, args=(split_sents[i], q_text, q_result))
        pipelines[i]['Sender'].append(p_sender)
        '''

    print('-'*40)
    print('Start Process...')
    for i in range(num_pipelines):
        for proc in pipelines[i]['Process']:
            proc.start()
    
    print(' Time split '.center(40,'-'))

    # ----- 以上时间算为加载用时 -----
    loadstime = (time.time() - start)*1000
    # ----- 开始计时 -----
    start = time.time()
    print('Start Sender...')
    for i in range(num_pipelines):
        #for proc in pipelines[i]['Process']:
        #    proc.start()
        
        print('Create sender process...')
        p_sender = mp.Process(target=send_dat, args=(split_sents[i], q_text, q_result))
        pipelines[i]['Sender'].append(p_sender)
        p_sender.start()

        #p = pipelines[i]['Sender'][0]
        #p.start()

    # 这里计算GPU内存占用
    memory = GPU_memory(0)
    print('Sender join...')
    for i in range(num_pipelines):
        p = pipelines[i]['Sender'][0]
        #p.start()
        p.join()

    predict_time = (time.time() - start)*1000
    # ----- 以上时间算为计算用时 -----
    
    avetime = predict_time/total_sent

    print('加载模型用时:%f 毫秒' % loadstime )
    print('Used Memory:%d MB'%memory)
    print('预测总计用时:%f 毫秒' % predict_time )
    print('预测单句用时:%f 毫秒' % avetime )

    result = 'DoublePipelineSplit', total_sent, memory, loadstime, predict_time, avetime
    # 追加到日志文件
    savetofile(result, logfile)

    # terminate
    for i in range(num_pipelines):
        for proc in pipelines[i]['Process']:
            proc.terminate()

