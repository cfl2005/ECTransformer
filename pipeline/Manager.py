# !/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# @Time     : 2020/9/29 
# @Author   : xiaoshan.zhang
# @Emial    : zxssdu@yeah.net
# @File     : Manager.py
# @Software : PyCharm
# ------------------------------------------------------------------------


class DepPipeLineManager(object):

    def __init__(self):
        self.pipeline_dict = {}

    def regist(self, pipeline_name, pipeline):
        """
        注册 pipeline
        :param pipeline_name:
        :param pipeline:
        :return:
        """
        self.pipeline_dict[pipeline_name] = pipeline

    def unregist(self, pipeline_name):
        if pipeline_name in self.pipeline_dict:
            del self.pipeline_dict[pipeline_name]

    def process(self, pipeline_name, input):
        self.pipeline_dict[pipeline_name].process(input)