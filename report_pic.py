#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import sys
import os
import re
import json
import time
import matplotlib.pyplot as plt

'''
根据report记录画出曲线图
'''

def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''
# 保存文本信息到文件
def savetofile(txt, filename, encoding='utf-8', method='w'):
    pass
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt))
        return 1
    except :
        return 0

def load_report(filename):
    ret = []
    txt = readtxt(filename)
    if txt:
        for line in txt.splitlines():
            if line:
                js = json.loads(line)
                ret.append(js)
    return ret

# 画显存使用曲线图
def drawpic(dats):
        
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(figsize=(10,5))
    
    titles = []
    for dat in dats:
        y = dat['mem data']
        title = '%s_%s_batch%s' % (dat['name'], dat['total_sent'], dat['batch_size'])
        titles.append(title)
        x = range(len(y))
        plt.plot(x, y) #, marker='-'


    plt.ylabel('GPU Memory(Mbyte)')
    plt.xlabel('time(seconds)')
    plt.title('GPU Memory')
    plt.legend(titles, loc='lower center')
    plt.savefig('report/report_20210812.png')
    plt.show()

if __name__ == '__main__':
    pass
    filename = 'report/report_20210812.txt'
    jsdat = load_report (filename)
    print(len(jsdat))
    fields = 'name,total_sent,batch_size,loadstime,predict_time,avetime,mem_ave,mem_max'.split(',')
    reptxt = []
    reptxt.append('\t'.join(fields))
    for dat in jsdat:
        rep = [ str(dat[field]) for field in fields] 
        reptxt.append('\t'.join(rep))

    txt = '\n'.join(reptxt)
    print(txt)
    repfile = 'report/report_20210812.csv'
    savetofile(txt, repfile)
    drawpic(jsdat)




