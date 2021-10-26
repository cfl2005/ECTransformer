#!/usr/bin/env python3
#coding:utf-8

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
import ECTrans_config

import warnings
warnings.filterwarnings('ignore')

gblgebug = True

def debug(func):
    global gbldebug 
    def wrapTheFunction(gbldebug):
        if gbldebug:
           func()
     
    return wrapTheFunction

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

# 客户端结果数据接收者
def result_collector(ip, port_out, total, task_id, result_queue):
    print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    # 设置过滤器
    filter_title = "{\"client_id\":\"%s" % task_id
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
            if cid:
                cons = 'work #%d' % cid
                if cons in collecter_data.keys():
                    collecter_data[cons] += t_sents
                else:
                    collecter_data[cons] = t_sents
        '''

        # 判断总记录数    
        if total_result >= total: break

    # 显示统计结果
    if collecter_data: pprint.pprint(collecter_data)
#-----------------------------------------
# 客户端
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5544,
                port_out=5545):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = ECTrans_config.batch_size
        self.result = []
        self.total = 0
        self.collector = None
        self.encoder = None
    
    # 设置编码方法, 将样本编码成可输入模型的数据
    def set_encoder(self, fun):
        self.encoder = fun

    # 发送数据文件
    def send(self, datafile):
        if self.encoder is None:
            raise ValueError("请先用set_encode设置编码方法")

        # 准备数据
        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        
        # 切割数据包
        total = len(sentences)
        '''
        batch_size = self.batch_size
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)
        print('task_id:', task_id)
        print('total sample:', total)
        print('total_batch:', total_batch)
        '''

        self.total = len(sentences)

        # 生成随机任务号
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)

        # 创建 接收端进程
        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        # 准备发送数据
        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        
        # 开始计时
        start = time.time()
        print('开始计时...')

        '''
        # 开始Producer之前必须先启动resultcollector和consumer
        for i in tqdm(range(total_batch), ncols=80):
            txts = sentlist[i]
            # 样本编码，转成list才能序列化
            # batch_text = get_sample(txts).numpy().tolist()

            batch_text = self.encodefun(txts)
            wid = '%d_%d' % (task_id, i)
            work_message = {'tid':wid, 'texts': batch_text, 'length': self.batch_size}
            zmq_socket.send_json(work_message)
            # 接收返回消息
            message = zmq_socket.recv()
        '''

        # 一次发送全部数据
        txts = sentences
        # 样本编码，转成list才能序列化
        batch_text = self.encoder(txts)
        work_message = {'tid':task_id, 'texts': batch_text, 'length': total}
        zmq_socket.send_json(work_message)
        # 接收返回消息
        message = zmq_socket.recv()


        # 接收数据
        print('等待数据返回...')
        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
        
        # 结束收集进程
        self.collector.terminate()
        self.collector.join()

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
    sys.path.append("../../../")
    from translate import get_sample

    parser = argparse.ArgumentParser(description='ECTrans框架客户端')
    parser.add_argument('--datafile', type=str, default="data_100.txt", help='数据文件')
    args = parser.parse_args()
    datafile = args.datafile

    # 启动客户端
    # python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_100.txt
    # python3 ECTrans.py --cmd=client --batch_size=8 --datafile=data_20.txt

    print('正在启动客户端...')

    client = ECTrans_Client(ip='127.0.0.1',
                        port=5557,
                        port_out=5560,)

    # 设置编码器
    txt_encode = lambda x: get_sample(x).numpy().tolist()
    client.set_encoder(txt_encode)

    print('正在发送数据...')
    sents = client.send(datafile)
    '''
    # 打印结果
    for sent in sents:
        print(sent)            
    '''
    print('total results :%d' % len(sents))

