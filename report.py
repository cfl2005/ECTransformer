#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


'''
生成报告结果
'''
import os
import sys
#import json
import pandas as pd

# 读入文件
def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''

filename = 'report.txt'
# 读入数据转为DataFrame
# ('SingleProcess', 20, 3327, 6328, 17267, 863)

text = readtxt(filename)
columns = ['program','sents','memory','load_time','total_time','ave_time']
dats = []
for line in text.splitlines():
    rec = eval(line)
    dats.append(list(rec))

df = pd.DataFrame (dats, columns=columns)
# 保存数据
#df.to_csv('reprot_dat.csv', index=0)
# 分类统计平均值
#df.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, **kwargs)
gp = df.groupby(by=['program','sents']).mean()
gp = gp.applymap("{0:.02f}".format)
print('Reprot'.center(40,'-'))
print(str(gp))


if __name__ == '__main__':
    pass

