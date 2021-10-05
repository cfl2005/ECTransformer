#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import time
import zmq
import random
# import re
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
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')


def consumer(ip, port, port_out, batch_size):

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
        result = translate_batch(sentences, model, batch_size=batch_size)
        
        # 组合数据
        jsdat = {'id': wid, 'result':result, 'consumer':consumer_id,}
        consumer_sender.send_json(jsdat)

# 消费者：编码器
def consumer_encoder(ip, port, port_out, model_encoder):
    # 创建ID号，创建ZMQ 
    consumer_id = random.randrange(1000,9999)
    print("consumer_encoder ID: #%s %d ==> %d" % (consumer_id, port, port_out) )
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
        dat = data['texts']
        # 转为Tensor
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        # 模型编码
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        #print('consumer_encoder:', consumer_id, type(src_enc), type(src_mask))
        torch.cuda.empty_cache()

        encode_dat = [src_enc.cpu().numpy().tolist(), src_mask.cpu().numpy().tolist()]
        # 组合数据,并发送
        jsdat = {'id': wid, 
                'encode_dat':encode_dat, 
                'consumer_encoder': consumer_id}
        consumer_sender.send_json(jsdat)

# 消费者：解码器
def consumer_decoder(ip, port, port_out, model_decoder, model_generator):
    # 创建ID号，创建ZMQ 
    consumer_id = random.randrange(1000,9999)
    print("consumer_decoder ID: #%s %d ==> %d" % (consumer_id, port, port_out) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))

    # 循环处理
    while True:
        # 接收数据
        data = consumer_receiver.recv_json()
        wid = data['id']
        consumer_encoder = data['consumer_encoder']
        src_enc, src_mask = data['encode_dat']

        # 转成 Tensor
        src_enc_ = torch.Tensor(src_enc).to(config.device)
        src_mask_ = torch.Tensor(src_mask).to(config.device)
        torch.cuda.empty_cache()

        # 解码器模型解码
        translation = translate_decode_split(src_enc_, src_mask_, model_decoder,
                                                model_generator, use_beam=True)
        torch.cuda.empty_cache()
        # print('translation:',translation)
        # 组合数据,并发送
        jsdat = {'id': wid, 
                'result': translation,
                'consumer_encoder':consumer_encoder,
                'consumer_decoder':consumer_id,
                }
        consumer_sender.send_json(jsdat)

# 数据转发者
def trans_collector(ip, port, port_out):
    print("trans_collector: %d ==> %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    # send worker
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.bind("tcp://%s:%s"%(ip, port_out))
    while True:
        dat = receiver.recv_json()
        consumer_sender.send_json(dat)
        #print('trans_collector: send data')

# 数据收集发布者
def result_pub(ip, port, port_out):
    print("result publisher: %d ==> %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        # 统一发布，客户端分主题订阅;
        publisher.send_json(ret)
        # title = str(ret)[:25]
        # print('publish:%s' % title)
    
# 任务接收器
def server_req(ip, port, port_out):
    print("server_req: %d ==> %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        sender.send_json(ret)
        receiver.send('OK'.encode())
        # print('server request')

# 客户端数据接收者
def result_collector(ip, port_out, total, task_id, result_queue):
    print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    # 设置过滤器
    filter_title = "{\"id\":\"%s" % task_id
    receiver.setsockopt(zmq.SUBSCRIBE, filter_title.encode())
    # print('filter_title:', filter_title)
    
    collecter_data = {}
    total_result = 0
    while True:
        # 接收数据
        ret = receiver.recv_json()
        sents = ret['result']
        # 发送到客户端队列
        result_queue.put(sents)

        t_sents = len(sents)
        total_result += t_sents
        '''
        # 统计各个进程完成数量
        for name in ['consumer_encoder', 'consumer_decoder']:
            cid = ret.get(name)
            cons = 'work #%d' % cid
            if cons in collecter_data.keys():
                collecter_data[cons] += t_sents
            else:
                collecter_data[cons] = t_sents
        '''

        # 判断总记录数    
        if total_result >= total: break

    # 显示统计结果
    #pprint.pprint(collecter_data)
    return       

#-----------------------------------------
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5559,
                port_out=5560,
                workers=4):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        self.port_out_encoder = port + 200 
        self.port_in_decoder = port + 300 
        self.port_out_publisher = port_out + 100
        self.port_out = port_out
        self.workers = workers
        # self.batch_size = batch_size
        self.p_workers = []
        self.trans_work = None

    def start(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(1)
        mp.set_start_method('spawn')

        # 加载拆分后的模型
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
        torch.cuda.empty_cache()

        # 创建消费者
        self.p_workers = []
        for i in range(self.workers):
            p_encode = mp.Process(target=consumer_encoder, 
                                    args=(self.ip, self.port_task, 
                                            self.port_out_encoder, model_encoder))
            
            p_decode = mp.Process(target=consumer_decoder,
                                    args=(self.ip, 
                                        self.port_in_decoder, self.port_out_publisher, 
                                        model_decoder, model_generator))

            self.p_workers.append([p_encode, p_decode])

        # 启动 消费者
        #print('consumer start...')
        print('encoder start...')
        for i in range(self.workers):
            self.p_workers[i][0].start()
            # p_workers[i][1].start()
        
        '''
        '''
        print('decoder start....')
        for i in range(self.workers):
            self.p_workers[i][1].start()

        # 启动 数据转发者 进程
        self.trans_work = mp.Process(target=trans_collector, 
                                    args=(self.ip, self.port_out_encoder, self.port_in_decoder))
        self.trans_work.start()

        # 启动 数据收集发布者 进程 
        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_publisher, self.port_out))
        self.publisher.start()

        # 启动 任务接收者 进程 
        self.server = mp.Process(target=server_req, 
                                args=(self.ip, self.port, self.port_task))
        self.server.start()


    def stop(self):
        self.trans_work.terminate()
        self.trans_work.join()
        for i in range(self.workers):
            self.p_workers[i][0].terminate()
            self.p_workers[i][1].terminate()
            self.p_workers[i][0].join()
            self.p_workers[i][1].join()

    def __enter__(self):
        pass

    def __exit__(self):
        pass
        self.stop()

    def join(self):
        pass

#-----------------------------------------
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5544,
                port_out=5545,
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
        # 生成随机任务号
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        '''
        print('task_id:', task_id)
        print('total sample:', total)
        print('total_batch:', total_batch)
        '''
        self.total = total

        # 创建 接收端进程
        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        # 准备发送数据
        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        
        '''
        # 开始计时
        start = time.time()
        print('开始计时...')

        '''
        # msg = input('输入回车开始发送数据:')
        # 开始Producer之前必须先启动resultcollector和consumer
        #for i in tqdm(range(total_batch), ncols=80):
        for i in range(total_batch):
            txts = sentlist[i]
            # 样本编码，转成list才能序列化
            batch_text = get_sample(txts).numpy().tolist()
            wid = '%d_%d' % (task_id, i)
            work_message = {'id':wid, 'texts': batch_text}
            zmq_socket.send_json(work_message)
            # 接收返回消息
            message = zmq_socket.recv()

        # 接收数据
        #print('等待数据返回...')
        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
        
        # 结束收集进程
        self.collector.terminate()
        self.collector.join()

        '''
        predict_time = (time.time() - start)*1000
        avetime = predict_time/total
        print('预测总计用时:%f 毫秒' % predict_time )
        print('预测单句用时:%f 毫秒' % avetime )
        '''
        return result 
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECTrans 框架')
    parser.add_argument('--cmd', type=str, required=True, default="", help='启动方式: server, client')
    parser.add_argument('--workers', type=int, default=2, help='启动多少个pipeline')
    parser.add_argument('--datafile', type=str, default="data_100.txt", help='数据文件')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    args = parser.parse_args()

    cmd = args.cmd
    datafile = args.datafile
    workers = args.workers
    batch_size = args.batch_size

    if cmd=='server':
        # 启动服务端 
        # python3 ECTrans.py --cmd=server --workers=2
        print('正在启动服务端...')
        server = ECTrans_Server(ip='127.0.0.1',
                                port=5557,
                                port_out=5560,
                                workers=workers)
        server.start()

    if cmd=='client':
        # 启动客户端
        # python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_100.txt
        # python3 ECTrans.py --cmd=client --batch_size=8 --datafile=data_20.txt

        print('正在启动客户端...')

        client = ECTrans_Client(ip='127.0.0.1',
                            port=5557,
                            port_out=5560,
                            batch_size=batch_size)
        print('正在发送数据...')                            
        sents = client.send(datafile)
        '''
        # 打印结果
        for sent in sents:
            print(sent)            
        '''
        print('total results :%d' % len(sents))
