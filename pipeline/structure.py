# !/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# @Time     : 2020/9/29 
# @Author   : xiaoshan.zhang
# @Emial    : zxssdu@yeah.net
# @File     : structure.py
# @Software : PyCharm
# ------------------------------------------------------------------------


from collections import defaultdict

class DiGraph(object):
    """
    拓扑排序:
        任何无回路的顶点活动网（AOV网）N都可以做出拓扑序列：
          1. 从N中选出一个入度为0的顶点作为序列的下一顶点。
          2. 从N网中删除所选顶点及其所有的出边。
          3. 反复执行上面两个步骤，知道已经选出了图中的所有顶点，或者再也找不到入度为非0的顶点时算法结束。
          4. 如果剩下入度非0的顶点，就说明N中有回路，不存在拓扑排序。
        存在回路，意味着某些活动的开始要以其自己的完成作为先决条件，这种现象成为活动之间的死锁。
    """

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices


    def addEdge(self, u, v):
        if v is None or v == '':
            self.graph[u] = []
        else:
            self.graph[u].append(v)

    def toposort_helper(self, v, visited, stack):

        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.toposort_helper(i, visited, stack)

        stack.insert(0, v)

    def recursion_toposort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.toposort_helper(i, visited, stack)

        return stack

    def loop_toposort(self):
        """
        非递归版本的拓扑排序
        :return:
        """
        in_degrees = dict((u, 0) for u in self.graph)
        vertext_num = len(in_degrees)

        for u in self.graph:
            # 计算每个顶点的入度
            for v in self.graph[u]:
                # print("当前入度字典为: {}".format(in_degrees))
                in_degrees[v] += 1

        Q = [u for u in in_degrees if in_degrees[u] == 0]

        Seq = []

        while Q:
            # 默认从最后一个删除
            u = Q.pop()
            Seq.append(u)
            for v in self.graph[u]:
                # 移除其所有的指向
                in_degrees[v] -= 1

                # 在次筛选入度为0的定点
                if in_degrees[v] == 0:
                    Q.append(v)

        # 如果循环结束后存在非0入度的定点
        # 说明图中有环， 不存在拓扑结构
        if len(Seq) == vertext_num:
            return Seq

        else:
            print("Graph exists a cricle")

def main():
    # 测试拓扑排序效果
    # g = DiGraph(6)
    # g.addEdge(5, 2)
    # g.addEdge(5, 0)
    # g.addEdge(4, 0)
    # g.addEdge(4, 1)
    # g.addEdge(2, 3)
    # g.addEdge(3, 1)

    """
    G = {
        'a': 'bce',
        'b': 'd',
        'c': 'd',
        'd': '',
        'e': 'cd'
    }
    """

    g = DiGraph(5)
    g.addEdge('a', 'b')
    g.addEdge('a', 'c')
    g.addEdge('a', 'e')
    g.addEdge('b', 'd')
    g.addEdge('c', 'd')
    g.addEdge('e', 'c')
    g.addEdge('e', 'd')
    g.addEdge('d', None)

    print("当前图结构为: \n{}".format(g.graph))

    print("拓扑排序结果: ")
    stack = g.loop_toposort()
    print(stack)


# 非递归版本
def toposort(graph):
    in_degrees = dict((u, 0) for u in graph)
    vertex_num = len(in_degrees)

    for u in graph:
        # 计算每个定点的入度
        for v in graph[u]:
            in_degrees[v] += 1
    Q = [u for u in in_degrees if in_degrees[u] == 0]

    Seq = []

    while Q:
        # 默认从最后一个删除
        u = Q.pop()
        Seq.append(u)
        for v in graph[u]:
            in_degrees[v] -= 1    # 移除其所有的指向

            # 再次筛选入度为0的定点
            if in_degrees[v] == 0:
                Q.append(v)

    # 如果循环结束后存在非0入度的定点
    # 说明图中有环， 不存在拓扑结构
    if len(Seq) == vertex_num:
        return Seq
    else:
        print("exists a circle.")



def loop_toposort():
    """
    测试非递归版本的 拓扑排序
    :return:
    """
    G = {
        'a': 'bce',
        'b': 'd',
        'c': 'd',
        'd': '',
        'e': 'cd'
    }

    print(toposort(G))

if __name__ == "__main__":
    main()
    # loop_toposort()