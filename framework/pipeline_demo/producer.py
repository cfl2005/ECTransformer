#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import time
import zmq
import random
from tqdm import tqdm

import torch.multiprocessing as mp
from resultcollector import *
from consumer import *

def producer(ip, port):
    '''
    # 先创建 consumer
    # worker数量
    workers = 2

    # 创建 consumer
    p_workers = []
    for i in range(workers):
        p = mp.Process(target=consumer, args=('127.0.0.1', 5557, 5558) )
        p_workers.append(p)
 
    # 启动 consumer
    for i in range(workers):
        p_workers[i].start()
    '''

    #-----------------------------------------
    # 创建发送端
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://%s:%s"%(ip, port) )
    msg = input('按下回车开始发送数据:')

    # 创建 接收端进程 
    result = {}
    total = 1000
    collector = mp.Process(target=result_collector, args=('127.0.0.1', 5558, total, result) )
    collector.start()

    # 开始Producer之前必须先启动resultcollector和consumer
    for i in tqdm(range(2000), ncols=80):
        num = random.randrange(10,100)
        work_message = {'num' : num}
        zmq_socket.send_json(work_message)
        if num % 100 == 0:
            time.sleep(0.1)
    
    collector.join()
    print('result:', result)

if __name__ == '__main__':
    pass
    producer('127.0.0.1', 5557)
