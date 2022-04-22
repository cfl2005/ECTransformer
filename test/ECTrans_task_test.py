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

from ECTrans_task import *


'''
批量测试脚本，启动多个客户端，每个客户端启动不同的用例 

'''

def dotask(datafile, embedding=0, batch=0):
    #print('正在启动客户端...')
    client = ECTrans_Client(ip="127.0.0.1",
                        port=5588,
                        port_out=5560,
                        batch_size=1)

    # 开始计时
    # start = time.time()
    #print('正在发送数据...')
    if batch:
        sents = client.send_batch(datafile, embedding=embedding)
    else:
        sents = client.send(datafile, embedding=embedding)

    tid = client.task_id
    now = time.time()
    total = len(sents)
    print('task id:%s, total:%d, finished_time:%s' % (tid, total, now))

# 运行测试
def run_test(task_list, embedding=0, batch=0):
    '''
    ip="127.0.0.1"
    port=5588
    port_out=5560

    # 创建 时间接收端进程
    time_queue = mp.Queue()
    collector = mp.Process(target=time_collector, 
                                args=(ip, port_out, time_queue))
    collector.start()
    '''

    # 执行每一个任务组
    for tasks in task_list:
        logging.info(('批量测试任务:%s'%str(tasks)).center(60, '-'))
        # 开始计时
        start = time.time()
        task_process = []
        for num, fn in tasks:
            datfile = './report/data_%s.txt' % fn
            for i in range(num):
                # 创建任务进程 
                process = mp.Process(target=dotask, args=(datfile,embedding, batch))                 
                task_process.append(process)
    
        # 启动所有客户端
        for p in task_process:
            p.start()
            #p.join()
            #time.sleep(0.01)

        #for p in task_process:
        #    p.join()

        predict_time = (time.time() - start)*1000
        logging.info('测试总计用时:%.2f 毫秒' % predict_time)

        '''
        # 接收时间统计返回
        result = []
        while time_queue.qsize() > 0:
            ret = time_queue.get()
            result.append(ret)
            # if time_queue.qsize() ==0 :break;
        all_time = np.array(result).sum()
        #logging.info('result:%s' % result)
        logging.info('预测总计用时:%.2f 毫秒' % all_time)
        
        time.sleep(2)
        '''

    # 结束进程
    #collector.terminate()
    #collector.join()


'''
* 组包任务测试：每包发送64个样本 
'''
def test_package():
    # 任务列表，(客户端个数，"文件名")
    task_list = [
                    [(2, '64')],
                    [(4, '64')],
                    [(8, '64')],
                    [(16, '64')]
                ]

    logging.info('组包任务测试'.center(40,'='))
    run_test(task_list)

def test_single(embedding=0):
    # 任务列表，(客户端个数，"文件名")
    task_list = [[(1, '1k')]]

    logging.info('实时任务测试'.center(40,'='))
    run_test(task_list, embedding=embedding)


'''
1.实时任务优化情况 
配置情况： pipe 4通道配置批处理0，实时4
'''
def test_real(batch=0):
    # 任务列表，(客户端个数，"文件名")
    task_list = [
                    [(16, '8')],
                    [(32, '8')],
                    [(64, '8')],
                    [(128, '8')]
                ]
    '''
    task_list = [
                    [(16, '1k')],
                    [(32, '1k')],
                    [(64, '1k')],
                    [(128, '1k')]
                ]
    '''

    logging.info('实时任务测试'.center(40,'='))
    run_test(task_list, batch=batch)

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
    # 混合任务：4个128，8个32
    task_list = [
                    [(4, '128'), (8, '32')],
                    [(4, '128'), (24, '32')],
                    [(4, '128'), (48, '32')],
                ]
    task_list = [
                    [(4, '128'), (8, '32')],
                ]

    task_list = [[(8, '32'), (4, '128')]]
    task_list = [[(4, '128'), (1, '20'), (1, '100'), (1, '8'), (1, '200'), (1, '50')]]
    task_list = [[(3, '256'), (1, '16'), (1, '256'), (1, '128'), (2, '256'), (1, '16'), (1, '128')]]

    logging.info('混合任务测试'.center(40,'='))
    run_test(task_list)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECTrans_Task框架测试')
    parser.add_argument('--cmd', type=str, required=True, default="", help='启动方式: real, mix')
    parser.add_argument('--log', type=str, default="ECTrans_task_test.log", help='日志文件')
    parser.add_argument('--embedding', type=int, default=0, help='编码方式,默认=0服务端编码')
    parser.add_argument('--batch', type=int, default=0, help='组包模式,默认=0不组包')
    args = parser.parse_args()

    cmd = args.cmd
    embedding = args.embedding
    batch = args.batch
    log = args.log


    fmttxt = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    ft = os.path.splitext(log)
    log = '%s_%s%s' % (ft[0], fmttxt, ft[1])

    #################################################################################################
    # 指定日志
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', log),
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

    if cmd == 'single':
        test_single(embedding=embedding)
    if cmd == 'real':
        test_real(batch=batch)
    if cmd == 'package':
        test_package()
    if cmd == 'mix':
        test_mix()
    
    
