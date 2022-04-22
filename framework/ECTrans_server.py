#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import time
import zmq
import random
import json
import logging
import numpy as np
from tqdm import tqdm
import pprint
import torch
import torch.multiprocessing as mp

import utils
import config
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

from ECTrans_config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')
gblgebug = True

def debug(func):
    global gbldebug 
    def wrapTheFunction(gbldebug):
        if gbldebug:
           func()
     
    return wrapTheFunction

# 调度器算法类
class SmratRouter():
    def __init__(self, senders) -> None:
        # self.queuelist = queuelist
        self.total_queue = n 
        self.queue_index = 0
        self.senders = senders
        # 缓存器队列
        self.cache = mp.Queue()
        self.timestrap = time.time()
        self.cache_length = 0
        self.process = None

    def send_json(self, dat):
        '''根据智能算法投递任务
        '''
        isbatch = 0
        # 根据数据大小确定实时还是批量
        length = len(dat['texts'])
        if length >= ECTrans_config.batch_value: # >=256为批处理
            # 批量处理
            isbatch = 1
        else:
            # 实时处理
            isbatch = 0

        tid = dat['tid']
        sentences = dat['texts']
        #dat.update({"client_ids": [(tid,(0, length))]})

        # 切割数据包
        total = length
        batch_size = ECTrans_config.batch_size
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]
        total_batch = len(sentlist)

        for i in range(total_batch):
            batch_text = sentlist[i]
            blen = len(batch_text)
            wid = '%d_%d' % (tid, i)
            ids = [(wid, (0, blen))]
            dat_msg = {'client_ids':ids, 'texts': batch_text, 'dat_len': blen}
            if blen == batch_size:
                self.sender[isbatch].send_json(dat_msg)
            else:
                self.cache.put(dat_msg)  # 零星数据放入队列
                  
    def start(self, cache):
        def que_process(cache):
            '''定时进程 处理零星数据，进行组包
            '''
            batch_size = ECTrans_config.batch_size
            while 1:
                dat_msg = None
                cur_length = 0
                while 1:
                    if cache.size > 0:
                        dat = cache.get()
                        # 组合数据包
                        if cur_length ==0:
                            dat_msg = dat
                            cur_length = dat['dat_len']
                        else:
                            tid, (b,e) = dat['client_ids'][0]
                            texts = dat['texts']
                            dat_len = dat['dat_len']
                            cur_length += dat_len
                            ids = (tid, (b + cur_length, e + cur_length))
                            # 添加到包中
                            dat_msg['client_ids'].append(ids)
                            dat_msg['texts'].extend(texts)
                            dat_msg['dat_len'] = cur_length 
                    
                    if cache.size ==0 or cur_length >= batch_size : break
                
                    # 发送数据
                    if dat_msg:
                        print('发送组合包...', dat_msg['client_ids'])
                        self.sender[0].send_json(dat_msg)
                
                # 时间间隔
                time.sleep(ECTrans_config.time_windows)

        self.process = mp.Process(target=que_process, args=(self.cache,))
        self.process.start()

# 服务端: 任务接收者  client ==> port(REP) ==> port_task(PUSH)
def server_req(ip, port, port_task, workers):
    # @debug
    print("server_req: %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    # 创建两个PUSH 分别为 实时 和批量
    sender_0 = context.socket(zmq.PUSH)
    sender_0.bind("tcp://%s:%d"%(ip, port_task))
    sender_1 = context.socket(zmq.PUSH)
    sender_1.bind("tcp://%s:%d"%(ip, port_task+1))
    sender = [sender_0, sender_1]

    # 创建调度器
    smart_router = SmratRouter(sender, n=workers)
    smart_router.start()
    while True:
        # 接收数据
        ret = receiver.recv_json()
        # 数据拆包重新组合
        smart_router.send_json(ret)
        receiver.send('OK'.encode())

def proc_encode(ip, port_task, q_enc, model_encoder):
    """
    编码器
    """
    # 创建ID号，创建ZMQ 
    consumer_id = random.randrange(1000,9999)
    print("proc_encode ID: #%s %d ==> Queue" % (consumer_id, port_task) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port_task))

    while True:
        # 获取任务数据
        data = consumer_receiver.recv_json()
        tid = data['client_ids']
        dat = data['texts']
        # 转为Tensor            
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        # print('src_enc:',type(src_enc))
        # print('encoder:%s' % consumer_id)
        q_enc.put( (tid, src_enc, src_mask) )
        torch.cuda.empty_cache()

def proc_decode(q_enc, ip, port_out_pub, model_decoder, model_generator):
    """
    解码器 
    """
    # send work
    context = zmq.Context()
    # zmq_socket = context.socket(zmq.REQ)
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect("tcp://%s:%d"%(ip, port_out_pub))

    while True:
        # 接收数据
        tid, dat, src_mask = q_enc.get()
        torch.cuda.empty_cache()
        src_enc = dat.clone()
        del dat
        torch.cuda.empty_cache()
        translation = translate_decode_split(src_enc, src_mask, model_decoder, model_generator, use_beam=True)
        torch.cuda.empty_cache()
        result = {'client_ids':tid, 'result':translation}
        # print('result:', result)
        zmq_socket.send_json(result)
        # 接收返回消息
        # message = zmq_socket.recv()

# 结果数据发布者
def result_pub(ip, port_out_pub, port_out):
    print("result publisher: %d ==> %d " % (port_out_pub, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port_out_pub))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        # todo: 拆包后逐个发送
        client_ids = ret['client_ids']

        for ids in client_ids:
            # ('1630303041986551_16',(0,64))
            client_id, (b, e) = ids
            dat = ret['result'][b:e]
            packet = {'client_id': client_id, 'result':dat}

            # 统一发布，由客户端分主题订阅;
            publisher.send_json(packet)
            # print('publish dat')
            # title = str(ret)[:25]
            # print('publish:%s' % title)

#-----------------------------------------
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5559,
                port_out=5560):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        # self.port_out_encoder = port + 200 
        # self.port_in_decoder = port + 300 
        self.port_out_pub = port_out + 100
        self.port_out = port_out

        self.workers_real = ECTrans_config.realtime_pipelines
        self.workers_batch = ECTrans_config.batch_pipelines
        self.workers = self.workers_batch + self.workers_real
        
        self.queues = []
        self.p_workers = []
        self.server_work = None
        self.publisher = None

    def start(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(1)
        #　['fork', 'spawn', 'forkserver']
        mp.set_start_method('spawn')

        #　{'file_system', 'file_descriptor'}
        # mp.set_sharing_strategy('file_system')
        '''
        strategies = mp.get_all_sharing_strategies()
        print('get_all_sharing_strategies:', strategies )
        print('strategies:', mp.get_sharing_strategy() )
        methods = mp.get_all_start_methods()
        print('get_all_start_methods:', methods )
        print('method :', mp.get_start_method() )
        '''

        # 创建队列
        for i in range(self.workers):
            q_encoder = mp.Queue()
            self.queues.append(q_encoder)
        
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
            # 流水线进程
            if i < self.workers_real:
                port_task = self.port_task
            else:
                port_task = self.port_task + 1
                
            p_encode = mp.Process(target=proc_encode, 
                                    args=(self.ip, port_task,
                                        self.queues[i], model_encoder))

            p_decode = mp.Process(target=proc_decode, 
                                    args=(self.queues[i], 
                                    self.ip, self.port_out_pub,
                                    model_decoder, model_generator))

            self.p_workers.append ([p_encode, p_decode])

        # 启动 编码器解码器
        print('encoder start...')
        for i in range(self.workers):
            self.p_workers[i][0].start()

        print('decoder start....')
        for i in range(self.workers):
            self.p_workers[i][1].start()

        # 启动 数据收集发布者 进程  
        print('publisher start....')
        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_pub, self.port_out))
        self.publisher.start()

        # 启动 任务接收者 进程
        print('server_work start....')
        self.server_work = mp.Process(target=server_req, 
                                args=(self.ip, self.port, self.port_task, self.workers))
        self.server_work.start()

        print('ECTrans server ready....')

    def stop(self):
        # self.trans_work.terminate()
        # self.trans_work.join()
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

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='ECTrans框架服务端')
    parser.add_argument('--workers', type=int, default=2, help='启动多少个pipeline')
    args = parser.parse_args()
    workers = args.workers
    '''

    workers = ECTrans_config.total_pipelines
    # 启动服务端 
    # python3 ECTrans.py
    print('正在启动服务端...')
    server = ECTrans_Server(ip='127.0.0.1',
                            port=5557,
                            port_out=5560,
                            workers=workers)
    server.start()
