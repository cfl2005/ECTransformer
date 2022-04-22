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
from collections import OrderedDict

import utils
import config
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

import ECTrans_config
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

def que_process(ip, port_task, cache):
    '''定时进程 处理零星数据，进行组包
    '''
    print("que_process: connect PUSH %d==> Queue " % (port_task) )
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://%s:%d"%(ip, port_task))
    
    while 1:
        dat_msg = None
        cur_length = 0
        while 1:
            if cache.qsize() > 0:
                dat = cache.get()
                # 组合数据包
                if cur_length ==0:
                    dat_msg = dat
                    cur_length = dat['dat_len']
                else:
                    tid, (b,e) = dat['client_ids'][0]
                    texts = dat['texts']
                    dat_len = dat['dat_len']
                    ids = (tid, (b + cur_length, e + cur_length))
                    # 添加到包中
                    dat_msg['client_ids'].append(ids)
                    dat_msg['texts'].extend(texts)
                    dat_msg['dat_len'] = cur_length 
                    cur_length += dat_len
                
            if cache.qsize() ==0 or cur_length >= ECTrans_config.batch_size : break
        
        # 发送数据
        if dat_msg:
            # print('发送组合包...', dat_msg['client_ids'])
            sender.send_json(dat_msg)
            #print('发送组合包完成...')
        # 时间间隔
        time.sleep(ECTrans_config.time_windows/1000)

# 实时处理收集
def real_pub(ip, port, port_out):
    print("publisher: PULL %d ==> PUSH %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    publisher = context.socket(zmq.PUSH)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        ids = ret.get('client_ids', 'null')
        publisher.send_json(ret)
        #print('组合包转发完成:', ids)

# # 调度器算法类
class SmratRouter():
    def __init__(self, ip, port_task, batch_size, batch_value): # cache,  -> None
        self.queue_index = 0
        #self.senders = senders
        # 缓存器队列
        self.cache = mp.Queue()         #mp.Queue()   cache
        self.timestrap = time.time()
        self.cache_length = 0
        self.process = None
        self.ip = ip
        self.port_task = port_task

        self.batch_value = batch_value
        self.batch_size = batch_size

        # 创建两个PUSH 分别为 实时 和批量
        context = zmq.Context()
        self.sender_0 = context.socket(zmq.PUSH)
        self.sender_0.connect("tcp://%s:%d"%(ip, port_task+2))
        self.sender_1 = context.socket(zmq.PUSH)
        #self.sender_1.bind("tcp://%s:%d"%(ip, port_task+1))
        self.sender_1.connect("tcp://%s:%d"%(ip, port_task+1))
        self.senders = [self.sender_0, self.sender_1]


    def send_json(self, dat):
        '''根据智能算法投递任务
        '''
        isbatch = 0
        # 根据数据大小确定实时还是批量
        length = len(dat['texts'])
        # print('texts length:', length)
        if length >= self.batch_value:
            # 批量处理 
            isbatch = 1
        else:
            # 实时处理
            isbatch = 0
        # print("isbatch:", isbatch)
        tid = dat['tid']
        sentences = dat['texts']

        # 切割数据包
        # print('正在拆分数据包...')
        total = length
        batch_size = self.batch_size
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]
        total_batch = len(sentlist)
        # print('total_batch:', total_batch)

        for i in range(total_batch):
            batch_text = sentlist[i]
            blen = len(batch_text)
            wid = '%d_%d' % (tid, i)
            ids = [(wid, (0, blen))]
            # print('ids:', ids, 'blen:', blen)
            # print(blen == self.batch_size)
            dat_msg = {'client_ids':ids, 'texts': batch_text, 'dat_len': blen}
            if blen == self.batch_size:
                # print('send to port')
                self.senders[isbatch].send_json(dat_msg)
            else:
                # print('send to queue')
                self.cache.put(dat_msg)  # 零星数据放入队列
                  
    def start(self):
        pass 
        '''
        '''
        print('正在启动调度器处理进程...')
        self.process = mp.Process(target=que_process, 
                                args=(self.ip, self.port_task+2, self.cache))
        self.process.start()
        print('调度器处理进程已启动。')

def server_route(ip, port, port_back):
    '''
    '''
    print("server_route: ROUTER %d ==> %d " % (port, port_back) )
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind("tcp://%s:%d"%(ip, port))
    
    backend = context.socket(zmq.ROUTER)
    backend.bind("tcp://%s:%d"%(ip, port_back))

    frontend.setsockopt(zmq.RCVHWM, 100)
    backend.setsockopt(zmq.RCVHWM, 100)
     
    workers = OrderedDict()
    clients = {}
    msg_cache = []
    poll = zmq.Poller()

    poll.register(backend, zmq.POLLIN)
    poll.register(frontend, zmq.POLLIN)

    while True:
        socks = dict(poll.poll(10))
        now = time.time()
        # 接收后端消息
        if backend in socks and socks[backend] == zmq.POLLIN:
            # 接收后端地址、客户端地址、后端返回response  
            # ps: 此处的worker_addr, client_addr, reply均是bytes类型
            worker_addr, client_addr, response = backend.recv_multipart()
            # 把后端存入workers
            workers[worker_addr] = time.time()
            if client_addr in clients:
                # 如果客户端地址存在,把返回的response转发给客户端,并删除客户端
                frontend.send_multipart([client_addr, response])
                clients.pop(client_addr)
            else:
                # 客户端不存在
                #print('addr:', worker_addr, client_addr)
                pass
        # 处理所有未处理的消息
        while len(msg_cache) > 0 and len(workers) > 0:
            # 取出一个最近通信过的worker
            worker_addr, t = workers.popitem()
            # 判断是否心跳过期 过期则重新取worker
            if t - now > 1:
                continue
            msg = msg_cache.pop(0)
            # 转发缓存的消息
            backend.send_multipart([worker_addr, msg[0], msg[1]])

        # 接收前端消息
        if frontend in socks and socks[frontend] == zmq.POLLIN:
            # 获取客户端地址和请求内容  ps: 此处的client_addr, request均是bytes类型
            client_addr, request = frontend.recv_multipart()
            clients[client_addr] = 1
            while len(workers) > 0:
                # 取出一个最近通信过的worker
                worker_addr, t = workers.popitem()
                # 判断是否心跳过期 过期则重新取worker
                if t - now > 1:
                    continue
                # 转发消息
                backend.send_multipart([worker_addr, client_addr, request])
                break
            else:
                # while正常结束说明消息未被转发,存入缓存
                msg_cache.append([client_addr, request])
    
def server_worker(ip, port, port_task): #, cache
    print("server_worker: DEALER %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.DEALER)
    # 设置接收消息超时时间为1秒
    receiver.setsockopt(zmq.RCVTIMEO, 1000)
    receiver.connect("tcp://%s:%d"%(ip, port))
    # 发送心跳到broker注册worker
    receiver.send_multipart([b"heart", b""])

    smart_router = SmratRouter(ip, port_task, #cache,
                ECTrans_config.batch_size, ECTrans_config.batch_value)
    smart_router.start()

    while True:
        try:
            # 获取客户端地址和消息内容
            client_addr, message = receiver.recv_multipart()
        except Exception as e:
            # 超时 重新发送心跳
            #print(e)
            receiver.send_multipart([b"heart", b""])
            continue
        # 处理任务
        # print('client:', client_addr, type(message), len(message))
        jsdat = json.loads(message)
        smart_router.send_json(jsdat)
        # 返回response
        receiver.send_multipart([client_addr, b"world"])

# 服务端: 任务接收者  client ==> port(REP) ==> port_task(PUSH)
def server_req(ip, port, port_task):
    # @debug
    print("server_req: REP %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    smart_router = SmratRouter(ip, port_task, ECTrans_config.batch_size, ECTrans_config.batch_value)
    smart_router.start()
    while True:
        # 接收数据
        dat = receiver.recv_json()
        # print('收到数据包，送进调度器...')
        # 数据拆包重新组合
        smart_router.send_json(dat)
        #tid = dat['tid']
        #print('task id:', tid)
        receiver.send('OK'.encode())

def proc_encode(ip, port_task, q_enc, model_encoder):
    """
    编码器
    """
    # 创建ID号，创建ZMQ 
    consumer_id = random.randrange(1000,9999)
    print("proc_encode ID: #%s ==> PORT PULL:%d" % (consumer_id, port_task) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port_task))

    while True:
        
        # 获取任务数据
        data = consumer_receiver.recv_json()
        # 开始计时
        start = time.time()

        tid = data['client_ids']
        dat = data['texts']
        # 转为Tensor            
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        # print('src_enc:',type(src_enc))
        # print('encoder:%s' % consumer_id)
        # 用时
        e_time = (time.time() - start)*1000

        q_enc.put( (tid, src_enc, src_mask, e_time) )
        torch.cuda.empty_cache()

def proc_decode(q_enc, ip, port_out_pub, model_decoder, model_generator):
    """
    解码器 
    """
    # send work
    consumer_id = random.randrange(1000,9999)
    print("proc_decode ID: #%s ==> PORT PUSH:%d" % (consumer_id, port_out_pub) )
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect("tcp://%s:%d"%(ip, port_out_pub))
    
    while True:
        # 接收数据
        tid, dat, src_mask, e_time = q_enc.get()
        # 开始计时
        start = 0
        start = time.time()

        torch.cuda.empty_cache()
        src_enc = dat.clone()
        del dat
        torch.cuda.empty_cache()
        translation = translate_decode_split(src_enc, src_mask, model_decoder, model_generator, use_beam=True)
        torch.cuda.empty_cache()
        # 用时
        d_time = (time.time() - start)*1000 #+ e_time

        result = {'client_ids':tid, 'result':translation, 'utime':(e_time, d_time)  } #(e_time, d_time)

        # print('client_ids:', tid)
        # print('result:', result)
        zmq_socket.send_json(result)

# 结果收集发布者
def result_pub(ip, port_out_pub, port_out):
    print("result publisher: %d ==> %d " % (port_out_pub, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port_out_pub))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        utime = ret['utime']
        publisher.send_json({'utime':utime})

        # todo: 拆包后逐个发送
        client_ids = ret['client_ids']
        # print('正在拆包:',  client_ids)

        for ids in client_ids:
            # ('1630303041986551_16',(0,64))
            client_id, (b, e) = ids
            dat = ret['result'][b:e]
            packet = {'client_id': client_id, 'result':dat}     #, 'utime':utime
            # print('packet client_id:', client_id)
            # 统一发布，由客户端分主题订阅;
            publisher.send_json(packet)

# 时间统计结果接收者
def time_collector(ip, port_out, time_queue):
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    # 设置过滤器
    filter_title = "{\"utime\":"
    receiver.setsockopt(zmq.SUBSCRIBE, filter_title.encode())
    
    collecter_data = {}
    total_result = 0
    while True:
        # 接收数据
        ret = receiver.recv_json()
        utime = ret['utime']
        #print('utime:', utime)
        # 发送到客户端队列
        time_queue.put(utime)
        

# 客户端结果数据接收者
def result_collector(ip, port_out, total, task_id, result_queue):
    #print("result_collector:  ==> %d " % (port_out) )
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
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5550,
                port_out=5560,
                realtime_pipelines=2,
                batch_pipelines=2
                ):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        # self.port_out_encoder = port + 200 
        # self.port_in_decoder = port + 300 
        self.port_out_pub = port_out + 100
        self.port_out = port_out

        #self.workers_real = ECTrans_config.realtime_pipelines
        #self.workers_batch = ECTrans_config.batch_pipelines
        self.workers_real = realtime_pipelines
        self.workers_batch = batch_pipelines
        self.workers = self.workers_batch + self.workers_real
        print('workers:', self.workers)

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
                # 0 实时
                port_task = self.port_task
            else:
                port_task = self.port_task + 5
                
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


        # 启动收集器进程
        print('real pub start....')
        self.real = mp.Process(target=real_pub, 
                                args=(self.ip, self.port_task+2, self.port_task))
        self.real.start()

        self.batch = mp.Process(target=real_pub, 
                                args=(self.ip, self.port_task+1, self.port_task+5))
        self.batch.start()


        # 启动 数据收集发布者 进程  
        print('publisher start....')
        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_pub, self.port_out))
        self.publisher.start()

        # 启动路由进程
        print('route start....')
        self.route = mp.Process(target=server_route, 
                                    args=(self.ip, self.port, self.port+1))
        self.route.start()


        '''
        print('正在启动调度器处理进程...')
        # 缓存器队列
        self.cache = mp.Queue()

        self.process = mp.Process(target=que_process, 
                                args=(self.ip, self.port_task+2, self.cache))
        self.process.start()
        print('调度器处理进程已启动。')
        '''

        # 启动 任务接收者 进程
        print('server_work start....')
        for i in range(3):
            self.server_work = mp.Process(target=server_worker,  #server_req
                                    args=(self.ip, self.port+1, self.port_task)) #, self.cache
            self.server_work.start()
        
        print('ECTrans server ready....')

    def stop(self):
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

# -----------------------------------------
# 客户端
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5550,
                port_out=5560):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = ECTrans_config.batch_size
        self.result = []
        self.total = 0
        self.collector = None
        self.encoder = None
        self.task_id = 0

    
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
        self.total = len(sentences)

        # 生成随机任务号
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        self.task_id = task_id

        # 创建 接收端进程
        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        # 准备发送数据
        context = zmq.Context()
        #zmq_socket = context.socket(zmq.REQ)
        zmq_socket = context.socket(zmq.DEALER)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        '''
        # 开始计时
        start = time.time()
        print('开始计时...')
        '''

        # 一次发送全部数据
        # 样本编码，转成list才能序列化
        batch_text = self.encoder(sentences)
        # print('batch_text:', type(batch_text))
        work_message = {'tid':task_id, 'texts': batch_text, 'length': total}
        zmq_socket.send_json(work_message)
        # 接收返回消息
        message = zmq_socket.recv()

        # 接收数据
        # print('等待数据返回...')
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
    parser.add_argument('--ip', type=str, default="127.0.0.1", help='ip')
    parser.add_argument('--port', type=int, default=5550, help='port')
    parser.add_argument('--port_out', type=int, default=5560, help='port_out')
    parser.add_argument('--realtime_pipelines', type=int, default=2, help='实时pipeline数量，最小为1')
    parser.add_argument('--batch_pipelines', type=int, default=2, help='批量pipeline数量')
    parser.add_argument('--datafile', type=str, default="report/data_100.txt", help='客户端发送的数据文件')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    args = parser.parse_args()

    cmd = args.cmd
    ip = args.ip
    port = args.port
    port_out = args.port_out

    realtime_pipelines = args.realtime_pipelines
    batch_pipelines = args.batch_pipelines

    datafile = args.datafile
    batch_size = args.batch_size

    if cmd=='server':
        # 启动服务端 
        print('正在启动服务端...')
        server = ECTrans_Server(ip=ip,
                                port=port,
                                port_out=port_out,
                                realtime_pipelines=realtime_pipelines,
                                batch_pipelines=batch_pipelines
                                )
        server.start()

    if cmd=='client':
        # 启动客户端
        # python3 ECTrans.py --cmd=client --datafile=report/data_100.txt
        # python3 ECTrans.py --cmd=client --datafile=report/data_20.txt

        print('正在启动客户端...')

        client = ECTrans_Client(ip=ip,
                            port=port,
                            port_out=port_out)

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

