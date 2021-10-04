#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
代码统计
'''
import os
import sys
import re
#-----------------------------------------
# 读入文件
def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''


# 使用生成器返回文件，避免长时间等待
# get all files in a folder, include subfolders
# fileExt: ['png','jpg','jpeg']
# return: 
#    return a Generator ,include all files , like ['./a/abc.txt','./b/bb.txt']
def getAllFiles_Generator (workpath, fileExt=[]):
    try:
        lstFiles = []
        lstFloders = []

        if os.path.isdir(workpath):
            if workpath[-1]!='/': workpath+='/'
            for dirname in os.listdir(workpath) :
                file_path = os.path.join(workpath, dirname)
                #file_path = workpath  + '/' + dirname
                if os.path.isfile(file_path):
                    if fileExt:
                        if dirname[dirname.rfind('.')+1:] in fileExt:
                           yield file_path
                    else:
                        yield file_path
                if os.path.isdir( file_path ):
                    yield from getAllFiles_Generator(file_path, fileExt)
        elif os.path.isfile(workpath):
            yield workpath
    except Exception as e :
        # print(e)
        pass

# 文件行数：包括所有行
def filelies(fname):
    texts = readtxt(fname)
    # 去掉空行
    texts = re.sub('([ 　\t]+)', '', texts)
    texts = re.sub('(\n\s+)', r"\n", texts)
    
    lines = len(texts.splitlines())

    return lines
# -----------------------------------------
listfile = 'list.txt'

filelists = readtxt(listfile)
# 过滤
files = [x.split(' ')[-1] for x in filelists.splitlines() if '.py' in x ]#
print(files[:8])
# 前8个为基础模型


# 记录行数：模型，框架，测试
lines = [0,0,0]
i = 0
# 遍历所有文件
for fn in getAllFiles_Generator('./', fileExt=['py']):
    fname = os.path.split(fn)[-1]
    #print(fname)
    if fname in files:
        ls = filelies(fn)
        print('%5d(lines) ==> File:%s' % (ls, fn))
        # 前8个为基础模型文件
        if fname in files[:8]:
            #print('model file')
            lines[0] += ls
        # 有'test'算测试文件
        elif 'test' in fn:
            lines[2] += ls
        else:
            lines[1] += ls
    i+=1
    #if i==10:break;
    
print('总文件数：%d, 总行数(模型，框架，测试):%s' % (i, lines))

if __name__ == '__main__':
    pass

