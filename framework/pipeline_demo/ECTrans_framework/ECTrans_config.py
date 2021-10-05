# EcTrans_framework 配置项

# 数据包组合大小
batch_size = 64

# 启动多少个pipeline数量；
total_pipelines = 4

# 批量计算的pipelines个数
batch_pipelines = 2

# 实时计算的pipelines个数
realtime_pipelines = 2

# 实时阈值：区分实时还是批量数据的阈值；大于等于阈值的数据量为批量数据；
batch_value = 256

# 数据处理时间窗口， 毫秒（不使用，移到 调度队列中）
time_windows = 500

