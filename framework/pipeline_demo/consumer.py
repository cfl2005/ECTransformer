#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import time
import zmq
import random

def consumer(ip, port, port_out):
    import random
    import time
    import zmq

    consumer_id = random.randrange(1000,9999)
    print("I am consumer #%s" % (consumer_id) )
    context = zmq.Context()

    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%d"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%d"%(ip, port_out))
    
    # 循环处理
    while True:
        work = consumer_receiver.recv_json()
        data = work['num']

        result = {'consumer':consumer_id, 'num':data}
        if data % 2 == 0: 
            consumer_sender.send_json(result)
            time.sleep(0.01)

if __name__ == '__main__':
    from multiprocess import Process

    # worker数量
    workers = 2
    if len(sys.argv)==2:
        workers = int(sys.argv[1])
 
    # 创建
    p_workers = []
    for i in range(workers):
        p = Process(target=consumer, args=('127.0.0.1', 5557, 5558) )
        p_workers.append(p)

    # 启动
    for i in range(workers):
        p_workers[i].start()




