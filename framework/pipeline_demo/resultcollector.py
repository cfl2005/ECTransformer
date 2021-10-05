#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import time
import zmq
import pprint
 
def result_collector(ip, port_out, total, result):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d" % (ip, port_out))

    collecter_data = {}
    for x in range(total):
        result = receiver.recv_json()
        # 这里可以再发送，分主题订阅；

        # 统计
        cons = 'work_%d' % result['consumer']
        if cons in collecter_data.keys():
            collecter_data[cons] += 1
        else:
            collecter_data[cons] = 1
    
    pprint.pprint(collecter_data)
    result = collecter_data

if __name__ == '__main__':
    pass
    result = {}
    result_collector('127.0.0.1', 5558, 1000, result)
    print('result:', result)
