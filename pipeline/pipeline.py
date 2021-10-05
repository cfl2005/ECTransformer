# !/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# @Time     : 2020/9/28 
# @Author   : xiaoshan.zhang
# @Emial    : zxssdu@yeah.net
# @File     : pipeline.py
# @Software : PyCharm
# @desc     : 并发型pipeline ， 强调pipe 的并发执行效率
#           java 实现的参考网址:
#             https://blog.csdn.net/tjy451592402/article/details/79459013
# ------------------------------------------------------------------------

"""
Pipe: 处理阶段的抽象， 负责对输入输出进行处理， 并将输出作为下一个阶段的输入。
      pipe 可以理解为 (输入、处理、输出) 三元组

init: 初始化当前处理阶段对外提供的服务。

shutdown: 关闭当前处理阶段，对外提供的服务。

setNextPipe: 设置当前处理阶段的下一个处理阶段。

ThreadPoolPipeDecorator:  基于线程池的Pipe 实现类， 主要作用是实现线程池去执行对各个输入元素的处理。

AbstractPipe:  Pipe 的抽象实现类。

process:  接收前一阶段的处理结果，作为输入， 并调用子类的doProcess 方法对元素进行处理，相应的处理结果
          会提交给下一个阶段进行处理

do_process: 留给子类实现的抽象方法

PipeContext:  对各个处理阶段的计算环境的抽象， 主要用于异常处理

Pipeline:  对 复合pipe的抽象， 一个Pipeline 实例可以包含多个pipe 实例。

addPipe:  向该Pipeline 实例中添加一个Pipe实例

SimplePipeline: 基于AbstractPipe 的 Pipeline 接口实现的一个简单类

PipelineBuilder :    pipeline 构造器， 用于从配置文件中加载构建 piepline

PipelineMananger: 管理多个Pipeline 的构建、销毁、执行

"""

import time
import random
#import Queue
import threading
from abc import ABCMeta, abstractmethod
from threading import Condition, Thread
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor



class CountDownLatch:
    """
      任务同步，用于同步异步任务，当注册了该同步锁的异步任务都执行完成后
      才释放锁。
    """

    def __init__(self, count):
        self.count = count
        self.condition = Condition()

    def await(self):
        try:
            self.condition.acquire()
            while self.count > 0:
                self.condition.wait()
        finally:
            self.condition.release()

    def countDown(self):
        try:
            self.condition.acquire()
            self.count -= 1
            self.condition.notifyAll()
        finally:
            self.condition.release()


class AbstractPipe(object):

    def __init__(self, pipe_name=None,  pipe_context=None):
        self.pipe_name = pipe_name
        self.next_pipe = None
        self.pipe_context = pipe_context

    def set_next(self, next_pipe):
        self.next_pipe = next_pipe

    def init(self, pipe_context):
        self.pipe_context = pipe_context

    def shut_down(self, timeout, time_unit):
        """
        关闭 pipe 执行的任务
        :param timeout:
        :param time_unit:
        :return:
        """
    def process(self, input):
        # try:
        out = self.do_process(input)

        if 'results' in self.pipe_context:
            self.pipe_context['results'][self.pipe_name] = out

        # 如果正确输出，并且当前正确定义了下一个pipe,调用下一个pipeline
        if out and self.next_pipe:
            print("当前 结果不为空， 下一个 pipe 不为 None: {}, 主动调用 下一个 pipe: {}".format(self.next_pipe, out))
            self.next_pipe.process(out)

    def do_process(self, input):
        raise NotImplementedError("Please implement do_process in inherit pipe class!")


class Function():
    __metaclass__ = ABCMeta

    def __init__(self, params={}, result={}, nlu_template=None,  nlg_template=None ):
        self.params = {}
        self.result = {}
        self.nlu_template = nlu_template
        self.nlg_tempalte = nlg_template

    def process(self, input):
        raise NotImplementedError("Please implement Function`s process logical")


    def gen_nlu_pattern(self):
        return self.nlu_template


    def gen_nlg_pattern(self):
        return self.nlg_tempalte

    def __call__(self, input):
        self.process(input)

class FunctionPipe(AbstractPipe):
    """
    Pipe 函数修饰类。
         调用内部封装的 函数类执行具体的逻辑
    """
    __metaclass__ =  ABCMeta

    def __init__(self, pipe_name, function):
        super(FunctionPipe, self).__init__(pipe_name=pipe_name)
        self.function = function

    @abstractmethod
    def do_process(self, inputs):
        """
        :param inputs:
        :return:
        """
        # 根据函数定义的参数列表，从context 中取出参数对应的值，
        kwargs = dict([(param_name, self.pipe_context[param_name]) \
                       for  param_name in self.function.params])

        # 传入 exec函数中
        result = self.function.execute(**kwargs)

        # 根据函数定义的返回参数列表，将处理结果放在 context 中
        for res_name in self.function.res_names:
            self.pipe_context[res_name] = result[res_name]

        # 返回 std_nlu 和 nlg语句

        std_nlu = None
        nlg = None

        return std_nlu , nlg

class Constraint(Function):
    """
    约束基类, 也是函数的一种

    直接解析往往比较困难，而且会不可避免地造成程序和语言的分歧。
    数据流的存在给了我们另一种选择：根据字面意思把引用解释成某种
    约束(Constraint)，再调用「解析」函数把符合约束的程序从数据
    流中找出来。
    """
    __metaclass__ =  ABCMeta

    def __init__(self,type_):
        self.type_ = type_

    def do_process(self, input):
        self.fit(input)

    @abstractmethod
    def fit(self,input):
        raise NotImplementedError("Please implement in inherit class!")



class ThreadPipeDecorator(AbstractPipe):
    """
    Pipe 的线程修饰类， 它不会维持一直存在的worker，而是任务到来时启动一个thread，
    这样， 内存压力会比较少，是标准的做法， 但是有线程切换开销。
    """
    def __init__(self, delegate_pipe, pool_executor):
        """
        :param delegate_pipe:
        :param pool_executor:
        """
        self.delegate_pipe = delegate_pipe
        self.thread_pool = pool_executor

    def init(self, pipe_context):
        """
        为业务对象 pipe 设置上下文
        :param pipe_context:
        :return:
        """
        self.delegate_pipe.init(pipe_context)

    def process(self, input):
        """
        注意 线程装饰器 的 process 函数不需要 调用 下一个 pipe， 由业务对象 pipe自己去调用
        :param input:
        :return:
        """
        print("当前 pipe thread recive input: {}".format(input))

        task = lambda input: self.delegate_pipe.process(input)

        self.thread_pool.submit(task, input)

        # 使用单线程 提交任务
        # thread = threading.Thread(target=task, args=[input,])
        # thread.setDaemon(True)
        # thread.start()


    def set_next(self, next_pipe):
        """
        为业务对象设置上下文
        :param next_pipe:
        :return:
        """
        self.delegate_pipe.set_next(next_pipe)


class WorkerPipeDecorator(AbstractPipe):
    """
    pipe 的线程池装饰类, 内部会维持一个一直运行的 worker， 无线程切换开销，但是在pipe个数多时，
    内存压力比较大

    说明: 使用线程池的时机:
        被 threadpool pipe 装饰器装饰的 pipe 不应该放入线程池中处理，因为线程池的同时运行的线程
        是有限的，但是 装饰器包装的 pipe 的线程的任务是放在 while 循环中的，不会主动结束。所以应该
        放在自由线程中。而 parallel pipe 的 sub pipe 应该放在线程池中，因为 sub pipe 只会执行一
        次，然后主动结束线程。
    """

    def __init__(self, delegate_pipe, pool_executor):
        """ """
        super(WorkerPipeDecorator, self).__init__()
        self.delegate_pipe = delegate_pipe
        self.thread_pool = pool_executor
        self.queue = Queue.Queue()          # 内部队列
        self.__active = False
        # 启动pipe worker
        self.start()


    def process(self, input):
        """
        :param input:
        :return:
        """
        # 这里 process 的作用是将 input 和 context 添加到 自己的queue中
        event = {"type": "pipe", "data": {
            "context": self.pipe_context,
            "input": input
        }}
        print("将输入 {} 封装成event: {}".format(input, event))
        self.queue.put(event)


    def start(self,):
        """
        启动 thread pipe
        :return:
        """
        self.__active = True
        print("将异步pipe 的运行状态设置为: {}".format(self.__active))
        def task():
            """
            当前线程
            :return:
            """
            print("启动 异步 pipe")
            while self.__active:
                event = self.queue.get(block=True, timeout=100)
                print("当前 parallel pipe 收到 event: {}".format(event))

                pipe_context =  event['data']["context"]
                input = event['data']['input']
                self.delegate_pipe.init(pipe_context)
                result = self.delegate_pipe.do_process(input)

                event["data"]['input'] = result

                # print("当前 pipe 的next pipe 为: {}, 封装的事件为: {}".format(self.next_pipe, event))
                # 将当前 pipe 处理完成的 result 传递给下一个 pipe
                self.next_pipe.queue.put(event)

        self.thread_pool.submit(fn=task)

    def shut_down(self):
        # 将当前的 thread pipe 设置为不活跃，关闭 pipe 的线程
        self.__active = False


class ParallelPipe(AbstractPipe):
    """
    说明使用线程池

    """

    def __init__(self, pipe_name=None,  pool=None):
        super(ParallelPipe, self).__init__(pipe_name=pipe_name)
        self.pipes = []
        self.count_down = None
        self.pool = pool if pool else ThreadPoolExecutor(max_workers=3)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def init(self, pipe_context):
        for pipe in self.pipes:
            pipe.init(pipe_context)
        self.count_down = CountDownLatch(len(self.pipes))

    def do_process(self, input):
        """
        并行执行 内部保存的各个子 pipe ， 所有pipe 执行完成才执行下游 pipe
        :param input:
        :return:
        """

        def task(pipe, input, count_down, callback=None):
            """
            将 pipe 的执行与 同步锁 count_down 包装在一起
            """
            # count_down.wait()
            result = pipe.do_process(input)
            # print("当前 parallel pipe 输出结果为: {}".format(result))
            if callback:
                callback(result)
            count_down.countDown()
            return result

        # results = []
        futures = []
        for pipe in self.pipes:
            # 每个pipe的输入使用 输入的副本
            input_cp = {"data": input['data']}
            future = self.pool.submit(task, pipe, input_cp, self.count_down)
            futures.append(future)

            # 下面是使用单线程的方法，注意这里使用了异步回调函数，收集任务结果。
            # thread = threading.Thread(target=task, args=(pipe, input_cp, self.count_down, lambda result:results.append(result)))
            # thread.setDaemon(True)
            # thread.start()

        # 进行同步，等待所有的子pipe的任务结束后收集结果
        self.count_down.await()
        # 从future 中取出结果
        results = [future.result() for future in futures]

        return results



class SimplePipeline(AbstractPipe):
    """
    简单 pipeline，
    """
    def __init__(self, pool_executor):
        self.thread_pool = pool_executor
        self.pipes = []

    def process(self, input):
        first_pipe = self.pipes[0]
        first_pipe.process(input)

    def init(self, pipe_context):
        """
        完成 pipe 链的链接， 以 pipe 执行上下文的注入
        :param pipe_context:
        :return:
        """
        prev_pipe = self
        self.pipe_context = pipe_context

        for pipe in self.pipes:
            prev_pipe.set_next(pipe)
            prev_pipe = pipe
            pipe.init(pipe_context)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def addAsThreadPoolBasedPipe(self, pipe):
        """
        将 pipe 按照多线程执行的方式添加到 pipeline 中
        :param pipe:
        :return:
        """
        self.add_pipe(ThreadPipeDecorator(pipe, self.thread_pool))

    def addAsWokerBasedPipe(self, pipe):
        """
        将 pipe 作为 woker 的任务添加到 pipeline中
        :param pipe:
        :return:
        """
        self.add_pipe(WorkerPipeDecorator(pipe, self.thread_pool))


class DependencyPipeline(AbstractPipe):
    """
    具有依赖关系的 Pipeline, 每个 pipe 执行前需要， 判断其依赖关系是否满足
    """
    def __init__(self, ):
        self.pipes = []

    def init(self, pipe_context):
        self.pipe_context = pipe_context

        for pipe in self.pipes:
            pipe.init(pipe_context)

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def dependency_check(self, pipe):
        """
        检查依赖关系
        从 context 查看 依赖的函数的 results 是否都输出了。
        :return:
        """
        is_check = True
        # 首先去除 pipe 依赖的任务
        dependencies = self.pipe_context['dependence'][pipe.pipe_name]

        if dependencies and len(dependencies) > 0:
            # 判断 依赖的是否返回结果
            for dep_pipename in dependencies:
                if self.pipe_context["results"][dep_pipename] is None:
                    print("pipe {} result is None".format(dep_pipename))
                    is_check = is_check and False

        return is_check

    def process(self, inputs):
        for pipe in self.pipes:
            is_check = self.dependency_check(pipe)
            if not is_check:
                print("Pipe {} dependency is not statisified, Please check it!")
            pipe.process(inputs)

    def reset(self):
        """
        重置当前pipeLine
        :return:
        """
        # 将当前 pipeline 的上下文中存储的result 置为 None
        for pipe_name in self.pipe_context['results']:
            self.pipe_context['results'][pipe_name] = None


# ---------------  pipe ----------------------------------------------------------

class DataTransformPipe(AbstractPipe):

    def __init__(self, indicator):
        # 先初始化父类
        super(DataTransformPipe, self).__init__()
        self.indicator = indicator

    def do_process(self, input):
        result = self.indicator + input['data']
        time.sleep(random.randint(1, 3))
        print("Data transform entit indicator: {}".format(self.indicator + input['data']))
        input["data"] = result
        return input


class MapPipe(AbstractPipe):
    """
    实现map功能的 pipe
    """
    def __init__(self, add_unit):
        super(MapPipe, self).__init__()
        self.add_unit = add_unit

    def do_process(self, input):

        input['data'] = input['data'] + self.add_unit
        print("Map pipe add unit: {}, result: {}".format(self.add_unit, input['data']))

        return input


class ReducePipe(AbstractPipe):
    """
    实现Reduce功能的Pipe
    """
    def __init__(self):
        super(ReducePipe, self).__init__()

    def do_process(self, input):
        print("Reduce pipe 接收到的内容为: {}".format(input))
        if not type(input) is list:
            inputs = [input]
        else:
            inputs = input

        sum = 0
        for input in inputs:
            sum += input['data']

        result = {"data": sum}

        print("Reduce Pipe result is {}".format(result))

        return result


def main():
    """
    测试 多线程执行方法
    :return:
    """
    pool = ThreadPoolExecutor(max_workers=20)
    simple_pipeline = SimplePipeline(pool_executor=pool)

    # 创建 pipe
    pipe_one = DataTransformPipe(indicator=1)
    pipe_two = DataTransformPipe(indicator=2)
    pipe_three = DataTransformPipe(indicator=3)
    pipe_four = DataTransformPipe(indicator=4)
    pipe_five = DataTransformPipe(indicator=5)


    # 测试 parallel pipe 的执行
    paral_pipe = ParallelPipe()
    for i in range(10):
        paral_pipe.add_pipe(MapPipe(i))

    reduce_pipe = ReducePipe()

    pipes = [pipe_one, pipe_two, pipe_three, pipe_four, pipe_five, paral_pipe, reduce_pipe]

    # for pipe in pipes:
    #     simple_pipeline.add_pipe(pipe)
    #
    # # 使用Pipe context 来初始化 simple_pipeline
    # print("使用单线程执行")
    # simple_pipeline.init(pipe_context={})
    # simple_pipeline.process(input={'data':10})


    # # 下面来验证 多线程下的执行
    for pipe in pipes:
        simple_pipeline.addAsThreadPoolBasedPipe(pipe)
        # simple_pipeline.addAsWokerBasedPipe(pipe)

    simple_pipeline.init(pipe_context={})

    print("使用多线程执行")
    # simple_pipeline.process(input={'data': 10})
    # simple_pipeline.process(input={'data': 20})
    # simple_pipeline.process(input={'data': 30})
    # simple_pipeline.process(input={'data': 40})
    # pool.shutdown(wait=True)

    for i in range(10):
        simple_pipeline.process(input={'data': 10 * i})


    # 注意这里需要保持主线程一致运行，否则 线程池也会 一起终止
    while True:
        time.sleep(2)
        print("主线程执行一次")


if __name__ == "__main__":
    main()
