#ECTransformer : A hierarchical GPU scheduling Framework for Transformer inference in edge computing 

Deep learning has been widely used in many applications. However, deep learning applications are generally computation-intensive and memory-intensive, so it is difficult to deploy on edge devices with limited resources.. Most of the current research focuses on distributed systems, job scheduling and model compression, but overlook the key issue of how to make more tasks run simultaneously on edge devices. In this paper, we propose ECTransformer, A hierarchical GPU scheduling Framework for Transformer inference in edge computing. In this paper, we analyze the characteristics of transformer model and the time cost on CPU and GPU. Guided by profiling, ECTransformer designs an edge-device partition method to reduce resource overhead on edge nodes.Ulteriorly, We also propose the deployment scheduling framework on the edge server side.By shared memory, scheduling queues and pipelines，the problems of high memory resource overhead and poor real-time performance of Transformer tasks on edge nodes are solved. Experiment results show that our proposed pipeline scheme saves 15-25 % of memory and speeds up 10 % to the conventional solution. When combined, ECTransformer can save up to 25% on GPU memory overhead and improve 5.1x processing efficiency.

# 开发总说明


## 总体说明


整个开发分为三个阶段：

    1、 pipeline阶段
        实现模型的分开加载，pipeline基础流程; 
        对基础流程进行优化；

    2、 框架阶段
        实现ECTrans_frame框架

    3、 框架调度器阶段
        实现ECTrans_framework框架加上智能调度器，分离实时与批量。

实验数据文件： `report/*.txt`

训练数据： `/data/`

分词器： `/tokenizer/`

模型目录： `/experiment/`

## Pipeline阶段

[开发文档] (doc/readme.md)

  - 实现模型的加载预测 `predict.py`
  - 实现对模型的拆分： `model_split.py`
  - 实现对模型分段加载，pipeline基础流程; 
  - 对基础流程进行优化: `predict_pipeline_split_n.py`

运行截图： `capture`

测试程序：`report.py`

批量测试脚本: `run_report.sh`

测试结果曲线图： `report_pic.py`

## 框架阶段

[开发文档](framework/readme.md)

主程序： `framework/pipline_demo/ECTrans.py`

截图目录 `framework/catpure`

测试程序： `framework/pipline_demo/ECTrans_test_batch.py`
测试结果：
  - 混合模式：`ECTrans_test_mix.log`, 
  - 实时模式： `ECTrans_test_real.log`
  
  `ECTrans_test.log`

## 框架调度器阶段

[开发文档](framework/readme.md)

[相关文档](framework/pipeline_demo/ECTrans_framework/doc)

主程序: `framework/pipeline_demo/ECTrans_framework/ECTrans_framework.py`

测试结果汇总: `framework/pipeline_demo/ECTrans_framework/doc/test_result.xlsx`


## 代码量统计

文件列表： `list.txt`

代码统计程序： `countlines.py`

统计结果： 

    包括空行： `总文件数：76, 总行数(模型，框架，测试):[1663, 4242, 558]`

    不含空行： `总文件数：76, 总行数(模型，框架，测试):[1365, 3544, 454]`
