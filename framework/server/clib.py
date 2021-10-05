#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import logging
import traceback
import pyssdb

SSDB_IP = '127.0.0.1'
SSDB_PORT = 8888
queue_in = 'queue_pipeline_in'
queue_out = 'queue_pipeline_out'

# 连接SSDB数据库
# SSDB_IP='127.0.0.1', SSDB_PORT=8888
def newSSDB(): 
    try:
        ssdb = pyssdb.Client(SSDB_IP, SSDB_PORT, None, 15 )
        logging.debug('open ssdb')
        return ssdb
    except Exception as e :
        logging.error('Eerror in newSSDB: '+ traceback.format_exc())
        return None


if __name__ == '__main__':
    pass

