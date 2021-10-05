# !/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# @Time     : 2020/9/29 
# @Author   : xiaoshan.zhang
# @Emial    : zxssdu@yeah.net
# @File     : manager_builder.py
# @Software : PyCharm
# @desc     : pipeline 管理器的builder
#             负责从指定的配置源加载 pipeline 以及 manager 的信息，并构建
# ------------------------------------------------------------------------

import os
import sys
import abc
import yaml
import imp
import inspect
from abc import ABCMeta, abstractmethod
from src.parallel_pipeline.structure import DiGraph
from src.parallel_pipeline.pipeline import AbstractPipe, DependencyPipeline
from src.parallel_pipeline.Manager import DepPipeLineManager


def import_file(filename):
    """
    使用指定的文件名，加载其包含的module
    :param filename:
    :return:
    """
    # 文件绝对路径
    path = os.path.abspath(os.path.dirname(filename))
    # 去除后缀的文件名
    name = os.path.splitext(os.path.basename(filename))[0]

    results = imp.find_module(name, [path])
    module = imp.load_module(name, results[0], results[1], results[2])

    return module


class TaskImporter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.tasks = {}

    def import_tasks(self):
        raise NotImplementedError("implement import_tasks function!")


class ModuleTaskImporter(TaskImporter):

    def __init__(self):
        super(ModuleTaskImporter, self).__init__()

    def import_tasks(self, modulefile):
        """
        从指定的 模块文件中加载任务
        :param modulefile:
        :return:
        """
        module = import_file(modulefile)

        # 获取指定 task 的所有子类
        sub_class_list = AbstractPipe.__subclasses__()

        for sub_class in sub_class_list:

            class_name = sub_class.__name__
            # print("当前的子类名称为: {}".format(class_name))
            # m_py = getattr(model_module, 'm')
            # 根据子类名称从m.py中获取该类
            has_subclass = hasattr(module, class_name)

            if has_subclass:
                task_class = getattr(module, class_name)
                # 实例化对象
                # obj = obj_class_name()
                # 调用print_name方法
                # getattr(obj, 'do_process')()
                self.tasks[class_name] = task_class
                # print("当前加载的子类: {}, 加载的对象为: {}".format(class_name, task_class))
            else:
                # print("当前子类: {}, 不在module{} 中".format(class_name, modulefile))
                continue

        return self



class ManagerBuilder(object):
    """
    管理器构建基础类
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.module_importer = ModuleTaskImporter()

    def build_controller(self, pipe_spec, tasks):
        pass

    def build_pipeline(self, spec):
        pipeline_name = spec['name']
        tasks_spec = spec['tasks']
        # 加载 pipeline 中的 依赖定义
        # print("当前定义的 pipeline 名称为: {}, 加载到的任务定义为: {}".format(
        #     pipeline_name, tasks_spec
        # ))


        pipe_dic = {}
        pipe_context = {"dependence": {},
                        "results": {}}
        for task_spec in tasks_spec:
            pipe_obj, pipe_dependence= self.process_task(task_spec)
            pipe_dic[pipe_obj.pipe_name] = pipe_obj
            # pipes.append(pipe_obj)
            # 初始化 依赖关系 和 结果
            pipe_context["dependence"][pipe_obj.pipe_name] = pipe_dependence
            pipe_context["results"][pipe_obj.pipe_name] = None

        # print("当前 pipe context 为: {}".format(pipe_context))

        # 0 使用依赖关系进行拓扑排序
        graph = DiGraph(len(pipe_dic))
        # 首先初始化各个节点的邻接列表为空
        for pipe in pipe_dic.values():
            graph.addEdge(pipe.pipe_name, None)
        # 根据依赖关系构建有向图
        for dest_pipename in pipe_context["dependence"]:
            # print("任务 {} 依赖的任务列表为: {}".format(dest_pipename, pipe_context["dependece"][dest_pipename]))
            if pipe_context["dependence"][dest_pipename] is None:
                continue
            for src_pipename in pipe_context["dependence"][dest_pipename]:
                graph.addEdge(src_pipename, dest_pipename)

        print("构建的依赖图结构为: \n {}".format(graph.graph))
        sorted_pipes = graph.loop_toposort()
        print("经过拓扑排序后的 pipes 顺序为: {}".format(sorted_pipes))
        pipes = [pipe_dic[pipe_name] for pipe_name in sorted_pipes]
        # print("新的pipes列表为: {}".format([pipe.to_dict() for pipe in pipes]))

        # 1. 根据依赖关系 初始化 pipe_context
        dep_pipeline = DependencyPipeline()

        for pipe in pipes:
            # 2. 使用pipe_obj 构建pipeline
            dep_pipeline.add_pipe(pipe)

        # 将运行时上下文注入到 dep_pipeline 中
        dep_pipeline.init(pipe_context)

        return pipeline_name, dep_pipeline


    def process_task(self, spec):
        """
        解析生成 task
        :param spec:
        :return:
        """
        # print("Task parser recieve spec : {}".format(spec))
        task_name      = spec['name']
        task_classname = spec['class']

        task_params    = {}
        for param in spec['params']:
            task_params.update(param)

        task_params.update({"pipe_name":task_name})

        task_depdencies = spec['dependence']

        # print("当前加载task的情况: task name: {}, class name: {}, params: {}, dependency：{}".format(
        #     task_name, task_classname, task_params, task_depdencies))

        # 1. 实例化 pipe
        pipe_obj =  self.importer.tasks[task_classname](**task_params)
        # print("当前加载的pipe_obj 为: {}".format(pipe_obj))

        # 2. 返回其依赖关系
        pipe_dependencies = task_depdencies

        return pipe_obj, pipe_dependencies

    def build_manager(self, spec):

        modules = spec["modules"]
        # 首先定义 importer , 将指定的模块先加载到内存中
        self.importer = ModuleTaskImporter()
        for module in modules:
            self.importer.import_tasks(module)

        # print(" manager 构建 接收到的 定义为: {}".format(spec))

        pipeline_manager = DepPipeLineManager()

        # 构建 Pipeline
        for pipeline_spec in spec['pipelines']:
            pipeline_name, dep_pipeline = self.build_pipeline(pipeline_spec)
            pipeline_manager.regist(pipeline_name, dep_pipeline)
            # 将构建好的 pipeline 注册到管理器中


        # print("当前得到的 pipe_manager 为: \n {}".format(pipeline_manager.pipeline_dict))

        return pipeline_manager


class PythonManagerBuilder(ManagerBuilder):

    def build_manager(self, filename):
        spec_module = import_file(filename)

        return super(PythonManagerBuilder, self).build_manager(spec_module)

class YamlManagerBuilder(ManagerBuilder):

    def build_manager(self, filename):

        with open(filename, 'r') as fin:
            spec = yaml.load(fin,Loader=yaml.FullLoader)

        # print("当前从yaml中加载的内容: {}".format(spec))

        return super(YamlManagerBuilder, self).build_manager(spec)



def main():
    """
    :return:
    """
    # file_name = "./tasks/task.py"
    # module = import_file(file_name)
    # print("加载的module 为: {}".format(module))
    # # dynamic_loadfromfile(file_name)
    #
    # # 测试加载指定 抽象类的子类
    # module_importer = ModuleTaskImporter()
    # module_importer.import_tasks(file_name)

    # 测试yaml 加载配置文件的方法
    yaml_file = "conf/dependcy_pipeline.yaml"
    manager_builder = YamlManagerBuilder()
    dep_pipeline_manager = manager_builder.build_manager(yaml_file)

    input = {"data":1}
    dep_pipeline_manager.process('TestPipeline', input)


if __name__ == "__main__":
    main()