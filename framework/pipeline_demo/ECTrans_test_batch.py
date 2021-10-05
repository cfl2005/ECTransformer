#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
import argparse
import os
import sys
import time
import torch.multiprocessing as mp
'''
import logging

from ECTrans import *


'''
批量测试脚本，启动多个客户端，每个客户端启动不同的用例 

'''

def dotask(datafile):
    #print('正在启动客户端...')
    # 开始计时
    start = time.time()
    client = ECTrans_Client(ip="127.0.0.1",
                        port=5557,
                        port_out=5560,
                        batch_size=8)

    #print('正在发送数据...')
    sents = client.send(datafile)
    #tid = client.task_id
    total = len(sents)
    predict_time = (time.time() - start)*1000
    avetime = predict_time/total
    result = []
    #result.append('-'*40 )
    #result.append('任务号:%s' % tid)
    result.append('返回条数:%d' % total)
    result.append('任务用时:%.2f ms' % predict_time)
    result.append('单句用时:%.2f ms' % avetime)
    logging.info('\t'.join(result) ) #+ '\n'
    

# 运行测试
def run_test(task_list):

    for tasks in task_list:
        logging.info(('批量测试任务:%s'%str(tasks)).center(60, '-'))
        # 开始计时
        start = time.time()
        task_process = []
        for num, fn in tasks:
            datfile = './report/data_%s.txt' % fn
            for i in range(num):
                # 创建任务进程 
                #process = mp.Process(target=dotask, args=(datfile,))                 
                #task_process.append(process)
                dotask(datfile)

        '''
        # 启动所有客户端
        for p in task_process:
            p.start()
        for p in task_process:
            p.join()
        '''

        predict_time = (time.time() - start)*1000
        logging.info('测试总计用时:%.2f 毫秒' % predict_time)
        time.sleep(2)

'''
1.实时任务优化情况 
配置情况： pipe 4通道配置批处理0，实时4
'''
def test_real():
    # 任务列表，(客户端个数，"文件名")
    task_list = [
                    [(16, '8')],
                    [(32, '8')],
                    [(64, '8')],
                    [(128, '8')]
                ]
    logging.info('实时任务测试'.center(40,'='))
    run_test(task_list)

'''
2.混合任务优化情况（全部批处理任务在Pipeline部分会详细对比）
服务器配置情况：pipe 4通道配置批处理2，实时2
'''
def test_mix():
    # 任务列表，(客户端个数，"文件名")
    '''
    task_list = [
                    [(4, '1k'), (8, '8')],
                    [(4, '1k'), (24, '8')],
                    [(4, '1k'), (48, '8')],
                ]
    '''
    task_list = [
                    [(4, '128'), (8, '32')],
                    [(4, '128'), (24, '32')],
                    [(4, '128'), (48, '32')],
                ]
    logging.info('混合任务测试'.center(40,'='))
    run_test(task_list)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECTrans框架测试')
    parser.add_argument('--cmd', type=str, required=True, default="", help='启动方式: real, mix')
    args = parser.parse_args()

    cmd = args.cmd

    #################################################################################################
    # 指定日志
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', 'ECTrans_test.log'  ),
                filemode='a'
                )
    #################################################################################################
    # 定义一个StreamHandler，将 INFO 级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    #formatter = logging.Formatter('[%(asctime)s]%(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################

    logging.info('ECTrans框架测试（无调度）'.center(40,'-'))
    
    if cmd == 'real':
        test_real()
    if cmd == 'mix':
        test_mix()
    
    
