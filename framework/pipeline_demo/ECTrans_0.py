#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import time
import zmq
import random
import re
import json
import logging
import numpy as np
from tqdm import tqdm
import pprint

import torch
#from torch.utils.data import DataLoader
#from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

import utils
import config
from train import translate
from model import make_model

from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')


def consumer(ip, port, port_out, batch_size):
    '''
    import random
    import time
    import zmq
    '''

    # 加载模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    #model.load_state_dict(torch.load(os.path.join('../../', config.model_path)))
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    torch.cuda.empty_cache()

    # 创建ID号，创建ZMQ 
    consumer_id = random.randrange(1000,9999)
    print("consumer ID: #%s" % (consumer_id) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))
    
    # 循环处理
    while True:
        # 获取任务数据
        data = consumer_receiver.recv_json()
        wid = data['id']
        sentences = data['texts']

        # 预测结果
        # batch_size = 8
        result = translate_batch(sentences, model, batch_size=batch_size)
        
        # 组合数据
        jsdat = {'id': wid, 'result':result, 'consumer':consumer_id,}
        consumer_sender.send_json(jsdat)

# 数据接收
def result_collector(ip, port_out, total, result):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port_out))

    collecter_data = {}
    total_result = 0
       
    #for x in range(total):
    while True:
        ret = receiver.recv_json()
        # 这里可以再发送，分主题订阅；

        sents = ret['result']
        result.extend(sents)
        t_sents = len(sents)
        wid = ret['id']
        total_result += t_sents
        
        # 统计各个任务完成数量
        cid = ret['consumer']
        cons = 'work_%d' % cid
        if cons in collecter_data.keys():
            collecter_data[cons] += t_sents
        else:
            collecter_data[cons] = t_sents
    
        if total_result >= total:
            break
    pprint.pprint(collecter_data)
    return result


#-----------------------------------------
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5588,
                port_out=5560,
                workers=4, 
                batch_size=8):
        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.workers = workers
        self.batch_size = batch_size

    def start(self):
        # 创建
        p_workers = []
        for i in range(self.workers):
            p = mp.Process(target=consumer, 
                            args=(self.ip, self.port, 
                                    self.port_out,
                                    self.batch_size))

            p_workers.append(p)

        # 启动
        for i in range(self.workers):
            p_workers[i].start()

    def stop(self):
        pass
        for i in range(self.workers):
            p_workers[i].terminate()

    def __enter__(self):
        pass

    def __exit__(self):
        pass
        for i in range(self.workers):
            p_workers[i].terminate()

#-----------------------------------------
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5588,
                port_out=5560,
                batch_size=8):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = batch_size
        self.result = []
        self.total = 0
        self.collector = None

    def send(self, datafile):
        # 准备数据
        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        batch_size = self.batch_size
        total = len(sentences)
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)
        work_id = random.randrange(1000, 9999)
        print('total:', total)
        print('total_batch:', total_batch)
        self.total = total

        # 创建 接收端进程
        result = []
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, result))
        self.collector.start()

        # 准备发送数据
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        zmq_socket.bind("tcp://%s:%d"%(self.ip, self.port) )

        # 开始计时
        start = time.time()

        # msg = input('输入回车开始发送数据:')
        # 开始Producer之前必须先启动resultcollector和consumer
        for i in tqdm(range(total_batch), ncols=80):
            wid = '%d_%d' % (work_id, i)
            work_message = {'id': wid, 'texts':sentlist[i]}
            zmq_socket.send_json(work_message)
            if i % 100 == 0:
                pass
                # time.sleep(0.1)

        # 接收数据
        self.collector.join()
        print('result:', len(result))
    
        predict_time = (time.time() - start)*1000
        avetime = predict_time/total
        print('预测总计用时:%f 毫秒' % predict_time )
        print('预测单句用时:%f 毫秒' % avetime )

        return result 
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    pass
    cmd = 'server'
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
    
    if cmd=='server':
        # 启动服务端
        print('正在启动服务端...')
        server = ECTrans_Server()
        server.start()

        #with ECTrans_Server() as server:
        #    server.start()

    if cmd=='client':
        # 启动客户端
        print('正在启动客户端...')
        datafile = 'data_100.txt'
        if len(sys.argv) >=3:
            datafile = sys.argv[2]


        '''
        with ECTrans_Client(ip='127.0.0.1',
                            port=5588,
                            port_out=5560) as client:
        '''
        client = ECTrans_Client(ip='127.0.0.1',
                            port=5588,
                            port_out=5560)
        print('正在发送数据...')                            
        rst = client.send(datafile)
        print('total results :%d' % len(rst))

    '''
    print()
    print('正在停止服务端...')
    server.stop()
    '''
