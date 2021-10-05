## 服务构架规划

### 方案一：

```
Client <===> HTTP Server <===> Queue <===> 模型预测服务
```
预热：
使用flask或者sanic启动WEB服务；
启动模型预测服务，可使用多进程或者pipeline; 

服务：
客户端发送请求数据到HTTP Server(websock 异步返回)
服务接收请求后将任务发送到队列Queue中（使用ssdb 做Queue)
队列Queue中的数据由模型预测进程进行处理；
预测完成的结果通过队列Queue异步传回HTTP Server; 

发送数据:
ws://192.168.40.11:8000/feed

接收返回数据:
ws://192.168.40.11:8001/rev

如果要搞大点，就上一个RabbitMQ,zeromq好像轻量点

完成了一个DEMO, 使用ssdb做queue

文件目录： `framework/server`

按顺序启动以下命令：

```
python server_rev.py
python server_pipeline.py
python server_feed.py
```


### 方案二：

改为使用ZMQ实现消息队列

```
应用端 <===> ECTrans_Client <===> ECTrans_server服务端
```

ECTrans_server服务端为：

```
PORT     <===> Producer(ZMQ.PUSH) <===> consumers(ZMQ.PULL, ZMQ.PUSH) <===> result collector
PORT_out <===>  result collector
```


### 实现ZMQ的DEMO 

文件目录： `framework/pipeline_demo`

在windows下即可测试，按顺序在三个窗口启动命令：

```
cd F:\project\ext_job\nmt_model\org\ChineseNMT\framework\pipeline_demo

python consumer.py 5
python resultcollector.py
python producer.py
```

发送1000条数据，消费进程5个；


### 用法包装
```
'''
# pip install ECTrans 可先不做安装包，使用目录
要先启动服务端；
python ECTrans_server.py --port=5588 \
			--port_out=5589 \
			--encode_model=model/encode_model.pth \
			--decode_model=model/decode_model.pth \
			--pipelines=4
'''

# 引用客户端
from ECTrans import ECTrans_Client

'''
# 分行写法
# 创建对象
client = ECTrans_Client(ip='127.0.0.1', port=5588)
# 预测
rst = client.send(list_text)
'''

# 推荐写法: 创建并预测
with ECTrans_Client(ip='127.0.0.1',
					port=5588,
					port_out=5589) as client:
	rst = client.send(list_text)

# array转换成list
result_txt = rst.tolist() 
```

### ZMQ+NMT Process DEMO

本地目录： `/framework/pipeline_demo/`

在远程服务器上模拟预测，使用直接加载整合模型

按顺序在三个窗口分别启动命令：

```
cd /mnt/workspace/ChineseNMT/framework/pipeline_demo

# 启动5个进程
python3 consumer_process.py 5 

# 总共100条数据
python3 resultcollector_nmt.py 100

# 文件名
python3 producer_nmt.py ../../data_100.txt
```

-----------------------------------------
## ECTrans 框架包装


文件名： `ECTrans.py`


用法：

```
# 启动服务端 
# python3 ECTrans.py --cmd=server --workers=2

# 启动客户端
# python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_100.txt

```

测试案例：

启动服务端 :

```
python3 ECTrans.py --cmd=server --workers=2
```

另开一个终端，启动客户端：
```
python3 ECTrans.py --cmd=client --datafile=data_100.txt
```

运行结果：
```
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --datafile=data_100.txt
正在启动客户端...
正在发送数据...
total: 100
total_batch: 13
开始计时...
100%|███████████████████████████████████████████| 13/13 [00:03<00:00,  3.80it/s]
等待数据返回...
收集数据...
{'work #5336': 8, 'work #6476': 40, 'work #8036': 44, 'work #9176': 8}
预测总计用时:8735.405684 毫秒
预测单句用时:87.354057 毫秒
total results :100
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --datafile=data_500.txt
正在启动客户端...
正在发送数据...
total: 500
total_batch: 63
开始计时...
100%|███████████████████████████████████████████| 63/63 [00:16<00:00,  3.73it/s]
等待数据返回...
收集数据...
{'work #5336': 32, 'work #6476': 220, 'work #8036': 216, 'work #9176': 32}
预测总计用时:39354.655266 毫秒
预测单句用时:78.709311 毫秒
total results :500
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --datafile=data_1k.txt
正在启动客户端...
正在发送数据...
total: 1000
total_batch: 125
开始计时...
100%|█████████████████████████████████████████| 125/125 [00:33<00:00,  3.78it/s]
等待数据返回...
收集数据...
{'work #5336': 328, 'work #6476': 168, 'work #8036': 168, 'work #9176': 336}
预测总计用时:76885.888100 毫秒
预测单句用时:76.885888 毫秒
total results :1000
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]#

```


不同参数的试验：
```
python3 ECTrans.py --cmd=client --batch_size=64 --datafile=data_2k.txt
python3 ECTrans.py --cmd=client --batch_size=64 --datafile=data_4k.txt
python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_2k.txt
python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_4k.txt

```

试验结果：
```
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --batch_size=64 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
total: 2000
total_batch: 32
开始计时...
100%|███████████████████████████████████████████| 32/32 [00:51<00:00,  1.60s/it]
等待数据返回...
收集数据...
{'work #5336': 512, 'work #6476': 464, 'work #8036': 512, 'work #9176': 512}
预测总计用时:124127.825499 毫秒
预测单句用时:62.063913 毫秒
total results :2000
-----------------------------------------
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --batch_size=64 --datafile=data_4k.txt
正在启动客户端...
正在发送数据...
total: 4000
total_batch: 63
开始计时...
100%|███████████████████████████████████████████| 63/63 [01:42<00:00,  1.63s/it]
等待数据返回...
收集数据...
{'work #5336': 992, 'work #6476': 1024, 'work #8036': 1024, 'work #9176': 960}
预测总计用时:235248.334408 毫秒
预测单句用时:58.812084 毫秒
total results :4000
-----------------------------------------
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
total: 2000
total_batch: 16
开始计时...
100%|███████████████████████████████████████████| 16/16 [00:50<00:00,  3.16s/it]
等待数据返回...
收集数据...
{'work #5336': 384, 'work #6476': 592, 'work #8036': 640, 'work #9176': 384}
预测总计用时:127796.109438 毫秒
预测单句用时:63.898055 毫秒
total results :2000
-----------------------------------------
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_4k.txt
正在启动客户端...
正在发送数据...
total: 4000
total_batch: 32
开始计时...
100%|███████████████████████████████████████████| 32/32 [01:39<00:00,  3.12s/it]
等待数据返回...
收集数据...
{'work #5336': 896, 'work #6476': 1056, 'work #8036': 1152, 'work #9176': 896}
预测总计用时:242990.249634 毫秒
预测单句用时:60.747562 毫秒
total results :4000
```


### 版本固定：version_202108251356

* 可多个客户端同时连接服务端发送任务；
* 各个客户端可返回各自发送的对应任务结果；


-----------------------------------------
## 基于Queue的PipeLine 框架

框架结构整理， 便于代码构建：

```
Client.sender  (connect port zmq.REQ)

    Server(
		==> server_req ( 								# 服务端任务接收器
				bind self.port zmq.REP,	
				bind self.port_task zmq.PUSH
				==> Smart Router (sender)  				# 智能路由器
				) 
		==> proc_encode (								# 编码器
				bind self.port_out_pub zmq.PULL, 		# 任务接收
				==> self.queues[i])
		==> proc_decode (								# 解码器
				self.queues[i],  						# 编码数据队列
				==> connect self.port_out_pub zmq.PUSH)	
		==> result_pub (								# 结果收集发布者
				bind self.port_out_pub zmq.PULL, 
				==>  bind self.port_out zmq.PUB)
	)

Client.result_collector  （connect port_out zmq.SUB)

```

服务端各子进程启动顺序：
 
 - result_pub 结果收集发布者
 - proc_encode 编码器
 - proc_decode 解码器
 - server_req 服务端任务接收器

### 测试

启动服务端：
```
python3 ECTrans_pipeline.py --cmd=server --workers=2
```

启动客户端， 不同参数的客户端试验：

```
python3 ECTrans_pipeline.py --cmd=client --batch_size=8 --datafile=data_100.txt
python3 ECTrans_pipeline.py --cmd=client --batch_size=64 --datafile=data_2k.txt
python3 ECTrans_pipeline.py --cmd=client --batch_size=64 --datafile=data_2k.txt
python3 ECTrans_pipeline.py --cmd=client --batch_size=64 --datafile=data_4k.txt
python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_2k.txt
python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_4k.txt
```

测试结果：
服务端：
```
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=server --workers=2
正在启动服务端...
get_all_sharing_strategies: {'file_descriptor', 'file_system'}
strategies: file_descriptor
get_all_start_methods: ['fork', 'spawn', 'forkserver']
method : spawn
Loading model...
encoder start...
decoder start....
proc_encode ID: #2296 5657 ==> Queue
proc_encode ID: #2861 5657 ==> Queue
publisher start....
server_work start....
ECTrans server ready....
result publisher: 5660 ==> 5560
server_req: 5557 ==> 5657

```

客户端：

```
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=8 --datafile=data_100.txt
正在启动客户端...
正在发送数据...
task_id: 1630303027709426
total sample: 100
total_batch: 13
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 13/13 [00:03<00:00,  3.93it/s]
等待数据返回...
预测总计用时:7986.178875 毫秒
预测单句用时:79.861789 毫秒
total results :100
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=64 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
task_id: 1630303041986551
total sample: 2000
total_batch: 32
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 32/32 [00:49<00:00,  1.54s/it]
等待数据返回...
预测总计用时:108162.456036 毫秒
预测单句用时:54.081228 毫秒
total results :2000
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
task_id: 1630303686960836
total sample: 2000
total_batch: 16
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 16/16 [00:48<00:00,  3.02s/it]
等待数据返回...
预测总计用时:108423.223734 毫秒
预测单句用时:54.211612 毫秒
total results :2000
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=64 --datafile=data_4k.txt
正在启动客户端...
正在发送数据...
task_id: 1630304408746144
total sample: 4000
total_batch: 63
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 63/63 [01:37<00:00,  1.55s/it]
等待数据返回...
预测总计用时:213677.032709 毫秒
预测单句用时:53.419258 毫秒
total results :4000
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_4k.txt
正在启动客户端...
正在发送数据...
task_id: 1630304672966113
total sample: 4000
total_batch: 32
开始计时...
result_collector:  ==> 5560
  3%|█▍                                          | 1/32 [00:03<01:34,  3.03s/it]
100%|███████████████████████████████████████████| 32/32 [01:35<00:00,  2.98s/it]
等待数据返回...
预测总计用时:217815.237999 毫秒
预测单句用时:54.453809 毫秒
total results :4000

```

同时两个客户端并发:

```
[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
task_id: 1630336662285832
total sample: 2000
total_batch: 16
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 16/16 [00:48<00:00,  3.06s/it]
等待数据返回...
预测总计用时:214356.784582 毫秒
预测单句用时:107.178392 毫秒
total results :2000

[root@iZ8vba5ftcg9rp4yipovm9Z ChineseNMT]# python3 ECTrans_pipeline.py --cmd=client --batch_size=128 --datafile=data_2k.txt
正在启动客户端...
正在发送数据...
task_id: 1630336660536438
total sample: 2000
total_batch: 16
开始计时...
result_collector:  ==> 5560
100%|███████████████████████████████████████████| 16/16 [00:49<00:00,  3.07s/it]
等待数据返回...
预测总计用时:213503.452778 毫秒
预测单句用时:106.751726 毫秒
total results :2000

```

-----------------------------------------


### 智能路由 算法描述

	1.	使用数据量和先后，分别使用不同权重，计算出优先级；
	2.	通过算法分析出机器负载情况，得出最大client并发数量，以及合适的batch size；
	3.	对worker进行分类，比如worker1 提供给数据量大/时间长的任务，排队进行，worker2 给 数据量小的进行，类似不同worker 打标 处理不同类型的任务；
	4.	上述完成后，将任务以batch为单位，均匀发给各个worker=PIPE，达到负载均衡的效果，避免热操作。



蔡丰龙  17:40:31
一级调度队列通过构建可调节的动态批处理模型（Batch size）实现对任务进行调度，
具体表现在：
（1）Batch size动态机制：根据模型类型、用户请求数和机器负载，动态调整适合框架的Batch size；
（2）Batch分类机制：将不同大小的batch分配给不同的二级调度队列进行计算;
（3）Batch打包机制：为了保证计算效率的最大化，将不足Batch size的个体进行打包，
	尽量构成以batch size为单位的计算单元；
（4）负载均衡机制：将（1-3）中的Batch以负载均衡的方式下发至二级调度队列，防止发生热读写的问题。
	（再总结一下，满载策略…冗余策略…）

蔡丰龙  17:43:32
1 server= 1worker
worker 下面 多个pipe 
（2）（4）根据不同情况分开讨论


蔡丰龙  23:46:48
二级调度架构：ECTransformer作为调度框架，具备两级调度机制：
一级调度为用户粒度调度，二级调度为用户内部任务调度。
一级调度队列通过构建可调节的动态批处理模型（Batch size）实现对任务进行调度，
具体表现在：
（1）Batch size动态机制：根据用户请求和机器负载，动态调整适合框架的Batch size；
（2）Batch分类机制：将不同类型的batch分配给不同的二级调度队列进行计算;
（3）Batch打包机制：为了保证计算效率的最大化，将不足Batch size的个体进行打包，
	尽量构成以batch size为单位的计算单元；
（4）负载均衡机制：将（1-3）中的Batch以负载均衡的方式下发至二级调度队列，防止发生热读写的问题。
	当满足请求处理条件情况下会进入二级处理流程，对请求进行处理，
	当不满足条件时，会在一级调度队列中进行等待。


需求分析拆解：
* 负载均衡机制
	构建配置文件及配置项，包括：启动worker数；确定各个worker的类型；
	实时阈值：区分实时还是批量数据的阈值；大于阈值的数据量为批量数据；

* 客户端直接把全部数据发送给服务端；由服务端负责对数据进行batch拆分及打包处理；

* 服务端： 智能调度器 对收到的数据进行batch打包处理；处理机制：打包成大小为64的包；
  每条数据均要标识客户端ID号，用于拆包时组合；
  用是否组合包标识来标记，标记为1时才需要进行拆包处理；

* 智能调度器 组合包处理机制
	组合包机制：数据不足时预留在队列中等待组合，
	相关设置：最大等待时间(例如100ms) 如超过等待时间没有新数据可组合，则直接组包；
	等待时间为内部变量，不要设置为配置项；

	任务ID记录： 
		client_ids:[('1630303041986551_16',(0,64))] 单个时为非组合包；
		client_ids:[('1630303041986551_16', (0,34)), ('1630303041987231_3', (34,64))] 多个值表示有组合的包
	序号标记：用索引号来记录各个数据对应的任务ID，在拆包时使用；
		record_ids: [(0,34), (34,64)] 
		表示：
			第1个任务ID对应记录为:dat[0:34] 0-33共34条
			第2个任务ID对应记录为:dat[34:64] 34-63共30条数据

	Tip: 组合包拆包测试样例： 多个客户端，多次发送数据量小于64的任务，会出现多个组合拆分的情况；


参考文档：  doc/PipeSwitch Fast Pipelined Context Switching.pdf
参考内容部分：`5 Implementation`


数据流格式汇总：

客户端发送： {'tid':"taskid_index", 'texts': batch_text}
服务端：
	调试器：解包


####  实验样例


对比mps，无任务调度场景：
基础：8 task mps vs [4 pipe  batch size=64]
1.实时任务优化情况 
配置情况： pipe 4通道配置批处理0，实时4

（1）16个任务 每一个都是8个，  16 * 8 表示 16个客户端，每个客户端发送8条数据
（2）32个任务 每一个都是8个，  32 * 8
（3）64个任务 每一个都是8个，  64 * 8 
（4）128个任务 每一个都是8个，128 * 8

2.混合任务优化情况（全部批处理任务在Pipeline部分会详细对比）
配置情况：pipe 4通道配置批处理2，实时2

（1）12个任务 4 task 1k个先来，后续8个8 size	4*1K +  8*8  4客户端发1K条，另8个客户端发8条数据
（2）28个任务，4 task 1k个先来，后续24个8 size	4*1K + 24*8
（3）52个任务，4 task 1k个先来，后续48个8 size	4*1K + 48*8


（1）12个任务 4 task 128个先来，后续8个8 size	
（2）28个任务，4 task 128个先来，后续24个8 size
（3）52个任务，4 task 128个先来，后续48个8 size


框架命令行：

启动服务端： 混合模式  实时与批量均启动2个pipeline 
python3 ECTrans_framework.py --cmd=server \
--ip=127.0.0.1 \
--port=5550 \
--port_out=5560  \
--realtime_pipelines=2  \
--batch_pipelines=2

启动服务端：实时模式 启动2个实时pipeline， 0个批量pipeline
python3 ECTrans_framework.py --cmd=server \
--ip=127.0.0.1 \
--port=5550 \
--port_out=5560  \
--realtime_pipelines=4  \
--batch_pipelines=0


启动客户端: 发送 100条数据用例文件
python3 ECTrans_framework.py --cmd=client \
--ip=127.0.0.1 \
--port=5550 \
--port_out=5560 \ 
--datafile=report/data_100.txt


启动批量测试任务：

```
python3 test_batch.py --cmd=mix
python3 test_batch.py --cmd=real
```


MPS:

```
开启
root@amax-2:~# export CUDA_VISIBLE_DEVICES=0
root@amax-2:~# nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
root@amax-2:~# nvidia-cuda-mps-control -d


关闭
root@amax-2:~# echo quit | nvidia-cuda-mps-control
root@amax-2:~# nvidia-smi -i 0 -c DEFAULT

```

https://blog.csdn.net/beckham999221/article/details/86644970

你先把你的都跑完, 然后我们看看时间

#### 不带调度的ECTrans 测试


启动服务端 :

```
python3 ECTrans.py --cmd=server --workers=4
```

启动批量测试任务：

```
python3 ECTrans_test_batch.py --cmd=mix
python3 ECTrans_test_batch.py --cmd=real
```

测试结果日志文件： 

	混合：ECTrans_test_mix.log
	实时：ECTrans_test_real.log


*** 测试结果汇总文件：  ***
`\framework\pipeline_demo\ECTrans_framework\doc\test_result.xlsx`




