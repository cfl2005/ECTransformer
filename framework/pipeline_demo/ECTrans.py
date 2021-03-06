#!/usr/bin/env python3
#coding:utf-8

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

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    #model.load_state_dict(torch.load(os.path.join('../../', config.model_path)))
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    torch.cuda.empty_cache()

    consumer_id = random.randrange(1000,9999)
    print("consumer ID: #%s" % (consumer_id) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))
    
    while True:
        data = consumer_receiver.recv_json()
        wid = data['id']
        sentences = data['texts']
        result = translate_batch(sentences, model, batch_size=batch_size)
        
        jsdat = {'id': wid, 'result':result, 'consumer':consumer_id,}
        consumer_sender.send_json(jsdat)

def consumer_encoder(ip, port, port_out, model_encoder):
    consumer_id = random.randrange(1000,9999)
    print("consumer_encoder ID: #%s %d ==> %d" % (consumer_id, port, port_out) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))
    
    while True:
        data = consumer_receiver.recv_json()
        wid = data['id']
        dat = data['texts']
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        #print('consumer_encoder:', consumer_id, type(src_enc), type(src_mask))
        torch.cuda.empty_cache()

        encode_dat = [src_enc.cpu().numpy().tolist(), src_mask.cpu().numpy().tolist()]
        jsdat = {'id': wid, 
                'encode_dat':encode_dat, 
                'consumer_encoder': consumer_id}
        consumer_sender.send_json(jsdat)

def consumer_decoder(ip, port, port_out, model_decoder, model_generator):
    consumer_id = random.randrange(1000,9999)
    print("consumer_decoder ID: #%s %d ==> %d" % (consumer_id, port, port_out) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))

    while True:
        data = consumer_receiver.recv_json()
        wid = data['id']
        consumer_encoder = data['consumer_encoder']
        src_enc, src_mask = data['encode_dat']

        src_enc_ = torch.Tensor(src_enc).to(config.device)
        src_mask_ = torch.Tensor(src_mask).to(config.device)
        torch.cuda.empty_cache()

        translation = translate_decode_split(src_enc_, src_mask_, model_decoder,
                                                model_generator, use_beam=True)
        torch.cuda.empty_cache()
        # print('translation:',translation)
        jsdat = {'id': wid, 
                'result': translation,
                'consumer_encoder':consumer_encoder,
                'consumer_decoder':consumer_id,
                }
        consumer_sender.send_json(jsdat)

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

def result_pub(ip, port, port_out):
    print("result publisher: %d ==> %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        publisher.send_json(ret)
        # title = str(ret)[:25]
        # print('publish:%s' % title)
    
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

def result_collector(ip, port_out, total, task_id, result_queue):
    print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    filter_title = "{\"id\":\"%s" % task_id
    receiver.setsockopt(zmq.SUBSCRIBE, filter_title.encode())
    # print('filter_title:', filter_title)
    
    collecter_data = {}
    total_result = 0
    while True:
        ret = receiver.recv_json()
        sents = ret['result']
        result_queue.put(sents)

        t_sents = len(sents)
        total_result += t_sents
     
        if total_result >= total: break

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
        
        model_encoder, model_decoder, model_generator = make_split_model(
                            config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)
        model_encoder.share_memory()
        model_decoder.share_memory()
        model_generator.share_memory()
       
        print('Loading model...')
        model_encoder.load_state_dict(torch.load(config.model_path_encoder))
        model_decoder.load_state_dict(torch.load(config.model_path_decoder))
        model_generator.load_state_dict(torch.load(config.model_path_generator))

        model_encoder.eval()
        model_decoder.eval()
        model_generator.eval()
        torch.cuda.empty_cache()

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

        self.trans_work = mp.Process(target=trans_collector, 
                                    args=(self.ip, self.port_out_encoder, self.port_in_decoder))
        self.trans_work.start()

        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_publisher, self.port_out))
        self.publisher.start()

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
        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        batch_size = self.batch_size
        total = len(sentences)
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        '''
        print('task_id:', task_id)
        print('total sample:', total)
        print('total_batch:', total_batch)
        '''
        self.total = total

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        
        for i in range(total_batch):
            txts = sentlist[i]
            batch_text = get_sample(txts).numpy().tolist()
            wid = '%d_%d' % (task_id, i)
            work_message = {'id':wid, 'texts': batch_text}
            zmq_socket.send_json(work_message)
            message = zmq_socket.recv()

        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
     
        return result 
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECTrans')
    parser.add_argument('--cmd', type=str, required=True, default="", help='start: server, client')
    parser.add_argument('--workers', type=int, default=2, help='pipeline num')
    parser.add_argument('--datafile', type=str, default="data_100.txt", help='file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    args = parser.parse_args()

    cmd = args.cmd
    datafile = args.datafile
    workers = args.workers
    batch_size = args.batch_size

    if cmd=='server':
        server = ECTrans_Server(ip='127.0.0.1',
                                port=5557,
                                port_out=5560,
                                workers=workers)
        server.start()

    if cmd=='client':

        print('start client...')

        client = ECTrans_Client(ip='127.0.0.1',
                            port=5557,
                            port_out=5560,
                            batch_size=batch_size)
        print('start send...')                            
        sents = client.send(datafile)
        '''
        for sent in sents:
            print(sent)            
        '''
        print('total results :%d' % len(sents))
